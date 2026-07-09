# References
# [1] Liu, Y., Yao, J., Cheng, Y., An, Y., Chen, X., Feng, S., Huang, Y., Shen, S.,
#     Zhang, R., Du, K., & Jiang, J. (2025). LMCache: An Efficient KV Cache Layer
#     for Enterprise-Scale LLM Inference. arXiv:2510.09665.
#
# This query implements the "cache population" step of the application-layer
# commitment cache. After each conversational exchange it scans the last few
# turns for explicit instructions or commitments and returns them as structured
# entries — mirroring the way LMCache extracts KV tensors from a completed
# inference pass and stores them for future reuse.

from dataclasses import dataclass

from synthetic_conversation_generation.data_models.character_card import CharacterCard
from synthetic_conversation_generation.data_models.commitment_cache import CommitmentEntry
from synthetic_conversation_generation.data_models.conversation import Conversation, ROLE
from synthetic_conversation_generation.llm_queries.llm_query import LLMQuery, ModelProvider

# Scan only the most recent exchange (2 turns) — we run after every exchange.
_SCAN_TURNS = 2


@dataclass
class CommitmentExtractionResult:
    entries: list[CommitmentEntry]


class CommitmentExtractionQuery(LLMQuery):
    """
    Scans the last conversational exchange and extracts any explicit instructions
    or commitments one character directed at the other.

    Implements the cache-population step in the application-layer analogue of
    LMCache (Liu et al., 2025): just as LMCache extracts attention KV tensors
    after a forward pass and persists them, this query extracts semantic
    commitments after each dialogue exchange and persists them in the
    CommitmentCache.

    A commitment is a direct, actionable instruction or agreement — e.g.
    "stop messaging me about this", "I'll send you the file by end of day",
    "don't ask me again". Vague statements of preference or emotional tone
    are NOT commitments and should be ignored.
    """

    def __init__(
        self,
        model_provider: ModelProvider,
        model_id: str,
        conversation: Conversation,
        character_a: CharacterCard,
        character_b: CharacterCard,
        turn_index: int,
    ):
        super().__init__(model_provider, model_id)
        self.conversation = conversation
        self.character_a = character_a
        self.character_b = character_b
        self.turn_index = turn_index

    def generate_prompt(self):
        recent = self.conversation.messages[-_SCAN_TURNS:] if self.conversation.messages else []

        lines = []
        for msg in recent:
            name = self.character_a.name if msg.role == ROLE.user else self.character_b.name
            lines.append(f"{name}: {msg.content}")
        exchange_text = "\n".join(lines) if lines else "(no messages yet)"

        names = [self.character_a.name, self.character_b.name]

        return f"""Extract explicit commitments or instructions from the following conversation exchange.

Characters: {names[0]} and {names[1]}

Exchange:
{exchange_text}

A commitment is a direct, actionable instruction or agreement — for example:
- "Stop messaging me about this" (instruction: recipient must stop)
- "Don't ask me again" (instruction: recipient must not repeat the question)
- "I'll send the file tomorrow" (promise: speaker will act)
- "Leave it, I'll handle it" (instruction: recipient should stand down)

Vague emotional statements, apologies, or personality traits are NOT commitments.
Only extract commitments that are explicit and actionable.

For each commitment found, state:
- speaker: who issued it (must be one of {names})
- recipient: who must honour it (must be one of {names})
- text: the commitment in one plain sentence

If no explicit commitments were made, return an empty list.
"""

    def response_schema(self):
        return {
            "type": "object",
            "properties": {
                "commitments": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "speaker": {"type": "string"},
                            "recipient": {"type": "string"},
                            "text": {"type": "string"},
                        },
                        "required": ["speaker", "recipient", "text"],
                        "additionalProperties": False,
                    }
                }
            },
            "required": ["commitments"],
            "additionalProperties": False,
        }

    def parse_response(self, json_response) -> CommitmentExtractionResult:
        valid_names = {self.character_a.name, self.character_b.name}
        entries = []
        for c in json_response.get("commitments", []):
            if c.get("speaker") in valid_names and c.get("recipient") in valid_names:
                entries.append(CommitmentEntry(
                    speaker=c["speaker"],
                    recipient=c["recipient"],
                    text=c["text"],
                    turn_index=self.turn_index,
                ))
        return CommitmentExtractionResult(entries=entries)

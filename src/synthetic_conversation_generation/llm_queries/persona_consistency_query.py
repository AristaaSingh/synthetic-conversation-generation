# References
# [1] Han, J., Koh, J., Seo, H., Chang, D., & Sohn, K. (2024).
#     PSYDIAL: Personality-based Synthetic Dialogue Generation using Large Language Models.
#     In Proceedings of LREC-COLING 2024, pp. 13321–13331.

from dataclasses import dataclass

from synthetic_conversation_generation.data_models.character_card import CharacterCard
from synthetic_conversation_generation.data_models.conversation import Conversation, ROLE
from synthetic_conversation_generation.llm_queries.llm_query import LLMQuery, ModelProvider


@dataclass
class PersonaConsistencyResult:
    is_consistent: bool
    reason: str


class PersonaConsistencyQuery(LLMQuery):
    """
    Post-generation filter: evaluates whether a candidate message is consistent
    with a character's personality given recent conversational context.

    Implements the dialogue filtering stage from Han et al. (2024) "PSYDIAL:
    Personality-based Synthetic Dialogue Generation using Large Language Models".
    PSYDIAL filters generated utterances through an LLM judge that checks
    profile alignment, personality alignment, and style alignment; utterances
    that fail are discarded and the generation is retried.

    This query is called after every CharacterMessageQuery. If it returns
    is_consistent=False, the pipeline regenerates the message (up to
    MAX_RETRIES attempts) before accepting a result. The filter is a
    programmatic loop in the generation architecture, not a prompt change.
    """

    _CONTEXT_TURNS = 4

    def __init__(
        self,
        model_provider: ModelProvider,
        model_id: str,
        character: CharacterCard,
        conversation: Conversation,
        candidate_message: str,
        is_character_a: bool,
        other_character_name: str,
    ):
        super().__init__(model_provider, model_id)
        self.character = character
        self.conversation = conversation
        self.candidate_message = candidate_message
        self.is_character_a = is_character_a
        self.other_character_name = other_character_name

    def generate_prompt(self):
        recent = self.conversation.messages[-self._CONTEXT_TURNS:] if self.conversation.messages else []

        sender_name = self.character.name
        other_name = self.other_character_name

        context_lines = []
        for msg in recent:
            name = sender_name if (msg.role == ROLE.user) == self.is_character_a else other_name
            context_lines.append(f"{name}: {msg.content}")
        context_text = "\n".join(context_lines) if context_lines else "(conversation just started)"

        return f"""Evaluate whether a text message is consistent with a character's personality.

Character being evaluated: {self.character.name}
Personality: {self.character.personality.strip()}

Recent messages leading up to this:
{context_text}

Candidate message from {self.character.name}:
"{self.candidate_message}"

Is this message consistent with {self.character.name}'s personality as described above?

Consider:
- Does the tone and register match what is described?
- Does the behaviour in this message match patterns described in the personality (e.g. brevity when impatient, dismissiveness, specific speech style)?
- Given what just happened in the conversation, is this how this character would actually respond?

Base your judgement strictly on the personality description. Do not penalise for being impolite or harsh if that is part of the character.
"""

    def response_schema(self):
        return {
            "type": "object",
            "properties": {
                "is_consistent": {
                    "type": "boolean",
                    "description": "True if the message authentically reflects the character's personality"
                },
                "reason": {
                    "type": "string",
                    "description": "One sentence explaining the assessment"
                }
            },
            "required": ["is_consistent", "reason"],
            "additionalProperties": False
        }

    def parse_response(self, json_response) -> PersonaConsistencyResult:
        return PersonaConsistencyResult(
            is_consistent=bool(json_response.get("is_consistent", True)),
            reason=json_response.get("reason", ""),
        )

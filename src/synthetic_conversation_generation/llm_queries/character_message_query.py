from datetime import datetime
import json

from synthetic_conversation_generation.data_models.conversation import Conversation, Message, ROLE
from synthetic_conversation_generation.data_models.character_card import CharacterCard
from synthetic_conversation_generation.data_models.scenario import Scenario
from synthetic_conversation_generation.llm_queries.llm_query import LLMQuery, ModelProvider

# After this many turns, drop the scenario from the prompt entirely.
# The conversation history is sufficient context — re-injecting the scenario
# description every turn pulls the model back to the opening scene.
_SCENARIO_SEED_TURNS = 4


def _format_gap(gap_minutes: float) -> str:
    if gap_minutes < 1:
        return "less than a minute"
    if gap_minutes < 60:
        return f"{int(gap_minutes)} minute{'s' if gap_minutes >= 2 else ''}"
    if gap_minutes < 1440:
        hours = gap_minutes / 60
        return f"{hours:.1f} hour{'s' if hours >= 2 else ''}"
    days = gap_minutes / 1440
    return f"{days:.1f} day{'s' if days >= 2 else ''}"


class CharacterMessageQuery(LLMQuery):
    """
    Generates the next message from a character in a two-person text conversation.

    The sender and receiver are CharacterCards. The Scenario seeds the opening
    context but is not repeated on every turn — the conversation history carries
    the narrative forward from there.
    """

    def __init__(
        self,
        model_provider: ModelProvider,
        model_id: str,
        conversation: Conversation,
        sender: CharacterCard,
        receiver: CharacterCard,
        scenario: Scenario,
        is_sender_character_a: bool,
        next_timestamp: datetime | None = None,
        gap_minutes: float | None = None,
    ):
        super().__init__(model_provider, model_id)
        self.conversation = conversation
        self.sender = sender
        self.receiver = receiver
        self.scenario = scenario
        self.is_sender_character_a = is_sender_character_a
        self.next_timestamp = next_timestamp or datetime.now()
        self.gap_minutes = gap_minutes

    def generate_prompt(self):
        turn_index = len(self.conversation.messages)

        temporal_context = ""
        if self.gap_minutes is not None:
            temporal_context = (
                f"\nTimestamp: {self.next_timestamp.strftime('%Y-%m-%d %H:%M')} "
                f"({_format_gap(self.gap_minutes)} since last message)\n"
            )

        # Scenario is only shown for the first few turns as a seed.
        # After that the conversation history is enough — repeating the scenario
        # description anchors the model to the opening scene indefinitely.
        if turn_index < _SCENARIO_SEED_TURNS:
            scenario_section = f"""
### How this conversation started
{self.scenario.description.strip()}

Relationship: {self.scenario.relationship.strip()}
"""
        else:
            scenario_section = ""

        history_lines = []
        for msg in self.conversation.messages:
            name = self.sender.name if msg.role == ROLE.user else self.receiver.name
            history_lines.append({
                "speaker": name,
                "timestamp": msg.timestamp.strftime("%Y-%m-%d %H:%M"),
                "message": msg.content,
            })

        return f"""You are {self.sender.name}. Write your next text message to {self.receiver.name}.

### You — {self.sender.name}
{self.sender.backstory.strip()}
{self.sender.personality.strip()}
{self.sender.description.strip()}

### {self.receiver.name}
{self.receiver.backstory.strip()}
{self.receiver.description.strip()}
{scenario_section}{temporal_context}
Write only the message text. Be completely faithful to who {self.sender.name} is. Let the conversation go wherever it naturally goes.

### Conversation so far
{json.dumps(history_lines, indent=2) if history_lines else "No messages yet."}
"""

    def response_schema(self):
        return {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": f"The next text message from {self.sender.name}"
                }
            },
            "required": ["text"],
            "additionalProperties": False
        }

    def parse_response(self, json_response) -> Message:
        content = (
            json_response.get("text")
            or json_response.get("message")
            or json_response.get("response")
            or json_response.get("content")
            or next(iter(json_response.values()), "")
        )
        return Message(
            message_id=len(self.conversation.messages),
            role=ROLE.user,
            content=content,
            timestamp=self.next_timestamp,
        )

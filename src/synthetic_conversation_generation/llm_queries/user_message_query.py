from dataclasses import asdict
from datetime import datetime
import json

from synthetic_conversation_generation.data_models.assistant import Assistant
from synthetic_conversation_generation.data_models.conversation import Conversation, Message, ROLE
from synthetic_conversation_generation.data_models.character_card import CharacterCard
from synthetic_conversation_generation.llm_queries.llm_query import LLMQuery, ModelProvider


def _format_gap(gap_minutes: float) -> str:
    """Convert a gap in minutes to a human-readable string."""
    if gap_minutes < 1:
        return "less than a minute"
    if gap_minutes < 60:
        return f"{int(gap_minutes)} minute{'s' if gap_minutes >= 2 else ''}"
    if gap_minutes < 1440:
        hours = gap_minutes / 60
        return f"{hours:.1f} hour{'s' if hours >= 2 else ''}"
    days = gap_minutes / 1440
    return f"{days:.1f} day{'s' if days >= 2 else ''}"


class UserMessageQuery(LLMQuery):

    def __init__(
        self,
        model_provider: ModelProvider,
        model_id: str,
        conversation: Conversation,
        user_persona: CharacterCard,
        assistant: Assistant,
        next_timestamp: datetime | None = None,
        gap_minutes: float | None = None,
    ):
        super().__init__(model_provider, model_id)
        self.conversation = conversation
        self.user_persona = user_persona
        self.assistant = assistant
        self.next_timestamp = next_timestamp or datetime.now()
        self.gap_minutes = gap_minutes

    def generate_prompt(self):
        temporal_context = ""
        if self.gap_minutes is not None:
            temporal_context = f"""
### Temporal Context
This message is sent at: {self.next_timestamp.strftime("%Y-%m-%d %H:%M")}
Time elapsed since last message: {_format_gap(self.gap_minutes)}
"""

        return f"""Generate the next message in this text message conversation between two people.

### Instructions
- Write only the raw message text — no speaker labels, no timestamps, no formatting
- Reflect the sender's personality, background, and communication style authentically
- Use natural SMS/text message style (brief, informal, contractions, occasional typos)
- The message content should be appropriate given how much time has passed since the last message
- Do not reference the time gap explicitly unless it would be completely natural to do so

### Sender Definition
{json.dumps(asdict(self.user_persona), indent=4)}

### Other Person in Conversation
{json.dumps(asdict(self.assistant), indent=4)}
{temporal_context}
### Conversation History
{json.dumps(self.conversation.prompt_format, indent=4)}
"""

    def response_schema(self):
        return {
            "type": "object",
            "properties": {
                "user_message": {
                    "type": "string",
                    "description": "The next text message from the sender"
                }
            },
            "required": ["user_message"],
            "additionalProperties": False
        }

    def parse_response(self, json_response) -> Message:
        return Message(
            message_id=len(self.conversation.messages),
            role=ROLE.user,
            content=json_response["user_message"],
            timestamp=self.next_timestamp,
        )

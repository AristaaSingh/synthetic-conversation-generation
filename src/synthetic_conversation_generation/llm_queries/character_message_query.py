from datetime import datetime
from typing import Optional

from synthetic_conversation_generation.data_models.conversation import Conversation, Message, ROLE
from synthetic_conversation_generation.data_models.character_card import CharacterCard
from synthetic_conversation_generation.data_models.world import World
from synthetic_conversation_generation.llm_queries.llm_query import LLMQuery, ModelProvider

# World context is shown for the first few turns only as a seed.
# After that the conversation history carries the narrative — re-injecting
# the world description every turn anchors the model to the opening context.
_WORLD_SEED_TURNS = 4


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
    """

    def __init__(
        self,
        model_provider: ModelProvider,
        model_id: str,
        conversation: Conversation,
        sender: CharacterCard,
        receiver: CharacterCard,
        world: World,
        is_sender_character_a: bool,
        next_timestamp: datetime | None = None,
        gap_minutes: float | None = None,
        state_summary: Optional[str] = None,
    ):
        super().__init__(model_provider, model_id)
        self.conversation = conversation
        self.sender = sender
        self.receiver = receiver
        self.world = world
        self.is_sender_character_a = is_sender_character_a
        self.next_timestamp = next_timestamp or datetime.now()
        self.gap_minutes = gap_minutes
        self.state_summary = state_summary

    def generate_prompt(self):
        turn_index = len(self.conversation.messages)

        temporal_context = ""
        if self.gap_minutes is not None:
            temporal_context = (
                f"\n({_format_gap(self.gap_minutes)} since last message, "
                f"{self.next_timestamp.strftime('%Y-%m-%d %H:%M')})\n"
            )

        # Sender and receiver roles in this world
        sender_role = self.world.character_a_role if self.is_sender_character_a else self.world.character_b_role
        receiver_role = self.world.character_b_role if self.is_sender_character_a else self.world.character_a_role

        sender_context = (
            self.world.character_a_context if self.is_sender_character_a else self.world.character_b_context
        ) or ""

        # World seed: only injected for the first few turns
        if turn_index < _WORLD_SEED_TURNS:
            world_section = (
                f"\nSetting: {self.world.setting.strip()}"
                f"\nYour role: {sender_role.strip()}"
                f"\n{self.receiver.name}'s role: {receiver_role.strip()}"
                f"\nRelationship: {self.world.relationship.strip()}"
            )
            if sender_context:
                world_section += f"\n{sender_context.strip()}"
            world_section += "\n"
        else:
            world_section = ""

        history_lines = []
        for msg in self.conversation.messages:
            name = self.sender.name if msg.role == ROLE.user else self.receiver.name
            history_lines.append(
                f"[{msg.timestamp.strftime('%Y-%m-%d %H:%M')}] {name}: {msg.content}"
            )
        history_text = "\n".join(history_lines) if history_lines else "(conversation just started)"

        state_section = f"\nContext: {self.state_summary}\n" if self.state_summary else ""

        return f"""You are {self.sender.name}. Write your next text message to {self.receiver.name}.

{self.sender.name}: {self.sender.personality.strip()}

{self.receiver.name}: {self.receiver.personality.strip()}
{world_section}{state_section}{temporal_context}
Write only the message text. Stay completely true to who {self.sender.name} is. This is a casual text conversation — write the way you would actually text, not an email.

Conversation so far:
{history_text}

Your message:"""

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

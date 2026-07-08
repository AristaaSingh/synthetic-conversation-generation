from synthetic_conversation_generation.data_models.conversation import Conversation
from synthetic_conversation_generation.data_models.character_card import CharacterCard
from synthetic_conversation_generation.llm_queries.llm_query import LLMQuery, ModelProvider


class ConversationCompletionQuery(LLMQuery):

    def __init__(
        self,
        model_provider: ModelProvider,
        model_id: str,
        conversation: Conversation,
        character_a: CharacterCard,
        character_b: CharacterCard,
    ):
        super().__init__(model_provider, model_id)
        self.conversation = conversation
        self.character_a = character_a
        self.character_b = character_b

    def generate_prompt(self):
        # Only examine the last few messages — sign-offs only matter at the tail end.
        # Showing the full history causes the model to see pending work items and
        # never return True, even when the last exchange was a clear goodbye.
        recent = self.conversation.messages[-6:]
        history_lines = []
        for msg in recent:
            name = self.character_a.name if msg.role.name == "user" else self.character_b.name
            history_lines.append(
                f"[{msg.timestamp.strftime('%Y-%m-%d %H:%M')}] {name}: {msg.content}"
            )

        return f"""Determine whether this text message exchange has reached a natural stopping point.

Look only at these most recent messages:
{chr(10).join(history_lines)}

Mark complete (True) if the LAST message contains a clear sign-off: goodbye, "talk later", "speak soon", "gotta go", "ttyl", "catch you then", "see you [time/day]", or an equivalent phrase that signals both people are done for now.

Do NOT consider whether there is outstanding work — people often agree to pick things up later and that is a valid ending. Focus only on whether the last message reads like a sign-off.

Err on the side of True if the last message has any goodbye-like quality.
"""

    def response_schema(self):
        return {
            "type": "object",
            "properties": {
                "is_complete": {
                    "type": "boolean",
                    "description": "True if the conversation has reached a natural pause or conclusion"
                }
            },
            "required": ["is_complete"],
            "additionalProperties": False
        }

    def parse_response(self, json_response):
        return json_response["is_complete"]

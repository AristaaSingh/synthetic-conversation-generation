import json

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
        history_lines = []
        for msg in self.conversation.messages:
            name = self.character_a.name if msg.role.name == "user" else self.character_b.name
            history_lines.append({
                "speaker": name,
                "timestamp": msg.timestamp.strftime("%Y-%m-%d %H:%M"),
                "message": msg.content,
            })

        return f"""Determine whether this text message conversation has ended.

### Characters
{self.character_a.name}: {self.character_a.summary}
{self.character_b.name}: {self.character_b.summary}

### Conversation
{json.dumps(history_lines, indent=4)}

### When to say the conversation is complete
Only mark complete if the last message contains an explicit sign-off: a goodbye, "talk later", "speak soon", "gotta go", "ttyl", or equivalent. An exchange that trails off or reaches a pause is NOT complete — people pick those back up.

A short conversation with only a few messages is almost never complete. If there is unresolved practical business (e.g. they are still coordinating something), it is not complete.

Err strongly on the side of False.
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

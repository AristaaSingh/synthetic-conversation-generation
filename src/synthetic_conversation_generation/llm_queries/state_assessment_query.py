from synthetic_conversation_generation.data_models.conversation import Conversation, ROLE
from synthetic_conversation_generation.data_models.character_card import CharacterCard
from synthetic_conversation_generation.data_models.conversation_state import ConversationState
from synthetic_conversation_generation.llm_queries.llm_query import LLMQuery, ModelProvider


class StateAssessmentQuery(LLMQuery):
    """
    Assesses the current state of the conversation after each exchange.

    Returns a ConversationState that drives:
    - Which Hawkes phase to use for upcoming inter-message timing
    - A running narrative summary passed as context to the next message generation

    Phase transitions are therefore event-driven (something happened in the
    conversation) rather than time-driven (a day threshold was crossed).
    """

    VALID_PHASES = {"early_contact", "escalation", "post_incident", "re_initiation"}

    def __init__(
        self,
        model_provider: ModelProvider,
        model_id: str,
        conversation: Conversation,
        character_a: CharacterCard,
        character_b: CharacterCard,
        previous_state: ConversationState,
    ):
        super().__init__(model_provider, model_id)
        self.conversation = conversation
        self.character_a = character_a
        self.character_b = character_b
        self.previous_state = previous_state

    def generate_prompt(self):
        history_lines = []
        for msg in self.conversation.messages:
            name = self.character_a.name if msg.role == ROLE.user else self.character_b.name
            history_lines.append(
                f"[{msg.timestamp.strftime('%Y-%m-%d %H:%M')}] {name}: {msg.content}"
            )

        return f"""You are analysing the current state of a text message conversation between two people.

{self.character_a.name}: {self.character_a.personality.strip()}
{self.character_b.name}: {self.character_b.personality.strip()}

Previous state: {self.previous_state.summary}
Phase: {self.previous_state.phase} | Tension: {self.previous_state.tension_level}/5

Conversation so far:
{chr(10).join(history_lines)}

### Your task
Assess the current state of the relationship and conversation. Determine:

1. **phase** — which of these best describes where the relationship is right now:
   - `early_contact`: tentative, polite, no significant tension yet
   - `escalation`: problematic behaviour is emerging or intensifying; tension is building
   - `post_incident`: a significant event has occurred (confrontation, withdrawal, something said that cannot be unsaid); one or both parties is pulling back
   - `re_initiation`: contact is being re-established after a period of distance or silence

   Only move to a later phase if the conversation content justifies it. Do not revert to an earlier phase.

2. **summary** — a 2-3 sentence description of where the relationship actually is right now. What has happened? What is the emotional undercurrent? What does each person actually feel, as distinct from what they are saying?

3. **tension_level** — an integer from 1 to 5:
   1 = completely neutral, nothing loaded
   2 = mild undercurrent, something slightly off
   3 = noticeable tension, one or both parties unsettled
   4 = significant friction, something has been said that landed badly
   5 = acute — confrontation, withdrawal, or crisis

4. **incident_occurred** — true if a significant relational event has happened (an explicit confrontation, a moment of clarity for either character, a withdrawal, something that marks a before/after). False if the conversation is still in ordinary flow.
"""

    def response_schema(self):
        return {
            "type": "object",
            "properties": {
                "phase": {
                    "type": "string",
                    "enum": ["early_contact", "escalation", "post_incident", "re_initiation"],
                    "description": "The current relationship phase"
                },
                "summary": {
                    "type": "string",
                    "description": "2-3 sentence narrative summary of the current relational dynamic"
                },
                "tension_level": {
                    "type": "integer",
                    "description": "Tension level from 1 (neutral) to 5 (acute conflict)"
                },
                "incident_occurred": {
                    "type": "boolean",
                    "description": "Whether a significant relational event has occurred"
                }
            },
            "required": ["phase", "summary", "tension_level", "incident_occurred"],
            "additionalProperties": False
        }

    def parse_response(self, json_response) -> ConversationState:
        phase = json_response.get("phase", self.previous_state.phase)
        if phase not in self.VALID_PHASES:
            phase = self.previous_state.phase

        return ConversationState(
            phase=phase,
            summary=json_response.get("summary", self.previous_state.summary),
            tension_level=max(1, min(5, int(json_response.get("tension_level", self.previous_state.tension_level)))),
            incident_occurred=json_response.get("incident_occurred", self.previous_state.incident_occurred),
        )

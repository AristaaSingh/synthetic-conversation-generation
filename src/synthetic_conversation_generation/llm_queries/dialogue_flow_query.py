# References
# [1] Bao, J., Wang, R., Wang, Y., Sun, A., Li, Y., Mi, F., & Xu, R. (2023).
#     A Synthetic Data Generation Framework for Grounded Dialogues.
#     In Proceedings of ACL 2023, pp. 10866–10882.
#
# [2] Morabito, R., Madhusudan, S., McDonald, T., & Emami, A. (2024).
#     STOP! Benchmarking Large Language Models with Sensitivity Testing on
#     Offensive Progressions. In Proceedings of EMNLP 2024, pp. 4221–4243.

from synthetic_conversation_generation.data_models.character_card import CharacterCard
from synthetic_conversation_generation.data_models.conversation_state import ConversationState
from synthetic_conversation_generation.data_models.dialogue_flow import Beat, DialogueFlow
from synthetic_conversation_generation.data_models.microaggression_taxonomy import is_valid
from synthetic_conversation_generation.data_models.world import World
from synthetic_conversation_generation.llm_queries.llm_query import LLMQuery, ModelProvider
from synthetic_conversation_generation.llm_queries.rolling_summary_query import RollingSummary


class DialogueFlowQuery(LLMQuery):
    """
    Pre-plans the beat sequence for a single conversation session.

    Implements the dialogue flow approach from Bao et al. (2023) "A Synthetic
    Data Generation Framework for Grounded Dialogues" (SynDG). Rather than
    letting the language model freely select a topic on every turn, this query
    runs once at the start of each session and produces an ordered list of beats.
    The generation model then realises each beat incrementally, receiving only
    the current beat as context — not the full planned sequence.

    Beat severity tiers follow the STOP (Morabito et al., 2024) offensive
    progression framework: each session's arc moves from the current tension
    level upward, encoding VAWG escalation structurally rather than through
    per-turn prompt adjustments.
    """

    def __init__(
        self,
        model_provider: ModelProvider,
        model_id: str,
        character_a: CharacterCard,
        character_b: CharacterCard,
        world: World,
        session_number: int,
        previous_state: ConversationState,
        rolling_summary: RollingSummary | None = None,
        exchange_budget: int = 6,
        min_beats: int = 3,
        max_beats: int = 6,
    ):
        super().__init__(model_provider, model_id)
        self.character_a = character_a
        self.character_b = character_b
        self.world = world
        self.session_number = session_number
        self.previous_state = previous_state
        self.rolling_summary = rolling_summary
        # The planner is given a BUDGET rather than a fixed beat count, and decides
        # both how many beats to use and how long each one needs. Previously the
        # count was hardcoded at 6 and every beat got exactly 2 turns, which was
        # (a) arbitrary and (b) arithmetically incompatible with the turn budget:
        # 6 beats x 2 turns x 6 sessions = 72 turns, against a max_turns of 60.
        self.exchange_budget = exchange_budget
        self.min_beats = min_beats
        self.max_beats = max_beats

    def generate_prompt(self):
        if self.rolling_summary:
            prior_context = (
                f"\nWhat has happened in previous sessions:\n"
                f"Events: {self.rolling_summary.events}\n"
                f"Relationship dynamic: {self.rolling_summary.dynamic}\n"
                f"Current state: tension {self.previous_state.tension_level}/5, "
                f"phase={self.previous_state.phase}\n"
                f"Summary: {self.previous_state.summary}\n"
            )
        elif self.session_number == 1:
            prior_context = "\nThis is the first session — no prior contact between these two.\n"
        else:
            prior_context = f"\nCurrent state: tension {self.previous_state.tension_level}/5, phase={self.previous_state.phase}\n"

        return f"""Plan the conversational arc for session {self.session_number} between two people.

{self.character_a.name}: {self.character_a.personality.strip()}

{self.character_b.name}: {self.character_b.personality.strip()}

Setting: {self.world.setting.strip()}
{self.character_a.name}'s role: {self.world.character_a_role.strip()}
{self.character_b.name}'s role: {self.world.character_b_role.strip()}
{prior_context}
You have a budget of about {self.exchange_budget} exchanges for this session (one exchange = one message from each person). Produce between {self.min_beats} and {self.max_beats} beats whose exchange counts sum to roughly that budget. Together they form a coherent session arc.

Each beat is defined on three axes: WHICH KIND of dynamic is in play (category), HOW INTENSE it is (severity), and HOW MUCH ROOM it needs (exchanges).

Microaggression categories available in this setting:
{self.world.category_definitions()}

Severity tiers (STOP framework):
1 = neutral — no problematic dynamic present
2 = subtle — a mild assumption, a slight dismissal, something slightly off
3 = noticeable — a pattern now visible across turns; one character unsettled
4 = significant — something said or done that lands badly; dynamic now explicit
5 = acute — confrontation, withdrawal, or a clear relational incident

How much room a beat needs (exchanges):
1 = a quick hand-off or logistical exchange — asked and answered, nothing more to say
2 = a normal back-and-forth with a little friction
3-4 = something lands badly and needs room to land, draw a reaction, and settle

Rules:
- Assign each beat the number of exchanges it genuinely needs. Do not give every beat the same number. Higher-severity beats generally need more room than trivial ones — a remark that stings cannot be delivered and resolved in one exchange.
- The exchange counts across all beats should sum to roughly {self.exchange_budget}
- Start at severity {self.previous_state.tension_level} or {max(1, self.previous_state.tension_level - 1)} (consistent with current tension)
- Escalate gradually — each beat may stay the same or rise by at most 1 severity point
- Each beat must name a concrete, real-world topic being discussed (e.g. "deployment pipeline error", "team meeting scheduling", "code review feedback")
- Topics must vary across beats — do not repeat the same subject
- Assign each beat a category from the list above, matching {self.character_b.name}'s behaviour in that beat. Vary the categories across the session — do not use the same one for every beat. Use "none" only for a severity-1 beat where no problematic dynamic is present.
- The description must specify {self.character_b.name}'s concrete behaviour in that beat, not a generic characterisation, and must be consistent with the category assigned to it
"""

    def response_schema(self):
        return {
            "type": "object",
            "properties": {
                "beats": {
                    "type": "array",
                    "minItems": self.min_beats,
                    "maxItems": self.max_beats,
                    "items": {
                        "type": "object",
                        "properties": {
                            "topic": {
                                "type": "string",
                                "description": "Concrete subject being discussed"
                            },
                            "category": {
                                "type": "string",
                                # Enum-constrained to the world's in-scope categories so the
                                # planner cannot invent one. "none" is permitted for neutral beats.
                                "enum": list(self.world.vawg_categories) + ["none"],
                                "description": "Which microaggression category is in play in this beat"
                            },
                            "severity": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": 5
                            },
                            "exchanges": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": 4,
                                "description": "How many back-and-forths this beat needs"
                            },
                            "description": {
                                "type": "string",
                                "description": f"Specific behaviour {self.character_b.name} exhibits in this beat"
                            }
                        },
                        "required": ["topic", "category", "severity", "exchanges", "description"],
                        "additionalProperties": False
                    }
                }
            },
            "required": ["beats"],
            "additionalProperties": False
        }

    def parse_response(self, json_response) -> DialogueFlow:
        beats = []
        for b in json_response["beats"]:
            # Defensive: Ollama does not hard-enforce enums the way the OpenAI/Anthropic
            # providers do, so validate rather than trust. An unrecognised or "none"
            # category becomes None (a neutral beat) instead of propagating a bad key.
            raw_category = str(b.get("category", "none")).strip()
            category = raw_category if is_valid(raw_category) else None

            beats.append(Beat(
                topic=b["topic"],
                severity=max(1, min(5, int(b["severity"]))),
                description=b["description"],
                category=category,
                # Clamp: Ollama does not hard-enforce numeric bounds. A beat of 0
                # exchanges would occupy no turns and be silently skipped.
                exchanges=max(1, min(4, int(b.get("exchanges", 1)))),
            ))
        return DialogueFlow(session_number=self.session_number, beats=beats)

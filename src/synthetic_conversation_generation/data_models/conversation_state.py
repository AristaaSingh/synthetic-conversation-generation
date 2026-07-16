from dataclasses import dataclass, field


@dataclass
class ConversationState:
    """
    Assessed state of the conversation after each exchange.

    Produced by StateAssessmentQuery and used to:
    - Drive Hawkes phase transitions (event-driven, not time-driven)
    - Provide rolling context to CharacterMessageQuery so each turn
      has a representation of where the relationship actually is
    """
    phase: str           # early_contact | escalation | post_incident | re_initiation
    summary: str         # brief narrative description of the current dynamic
    tension_level: int   # 1 (neutral) to 5 (acute conflict or crisis)
    incident_occurred: bool  # whether a significant event has happened yet

    # Canonical microaggression categories the assessor judges to be ACTUALLY
    # present, as distinct from those the dialogue-flow planner INTENDED. The gap
    # between intended (Beat.category) and realised (this field) is itself an
    # evaluation signal: it measures whether the generator delivered the plan.
    detected_categories: list[str] = field(default_factory=list)

    # Whether the conversation has reached a natural stopping point for now.
    # Previously judged by a separate ConversationCompletionQuery, which cost ~30
    # LLM calls per run and saw only the last 6 messages. Folded in here at zero
    # marginal cost: the assessor already reads the whole conversation and knows
    # the tension and phase, which is exactly the context that determines whether
    # a conversation would naturally stop. Session ends stay emergent — they are
    # not planned in advance, because a conversation's ending is a response to
    # what happened in it.
    session_ended: bool = False

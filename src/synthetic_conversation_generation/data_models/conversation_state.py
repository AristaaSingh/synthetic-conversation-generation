from dataclasses import dataclass


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

# References
# [1] Bao, J., Wang, R., Wang, Y., Sun, A., Li, Y., Mi, F., & Xu, R. (2023).
#     A Synthetic Data Generation Framework for Grounded Dialogues.
#     In Proceedings of ACL 2023, pp. 10866–10882.
#
# [2] Morabito, R., Madhusudan, S., McDonald, T., & Emami, A. (2024).
#     STOP! Benchmarking Large Language Models with Sensitivity Testing on
#     Offensive Progressions. In Proceedings of EMNLP 2024, pp. 4221–4243.

from dataclasses import dataclass, field


# Beat severity tiers are aligned with the STOP offensive progression scale [2].
@dataclass
class Beat:
    """
    One planned moment in a conversational arc.

    topic       — what the characters are texting about in this beat
    severity    — STOP-scale severity tier [2]:
                    1 = neutral; 2 = subtle; 3 = noticeable;
                    4 = significant; 5 = acute
    description — the relational dynamic or behaviour pattern active here
    """
    topic: str
    severity: int
    description: str


# Dialogue flow structure is adapted from SynDG [1]: a sequence of knowledge/topic
# pieces pre-planned before generation, realised one beat at a time.
@dataclass
class DialogueFlow:
    """
    Pre-planned sequence of beats for one session.

    Adapted from the dialogue flow concept in SynDG [1]: the pipeline
    determines the conversational trajectory before generation begins, then
    realises each beat incrementally. The generator receives one beat at a
    time, preventing topic lock and encoding the intended escalation arc
    structurally rather than through per-turn prompt adjustments.
    """
    session_number: int
    beats: list[Beat]
    _current_index: int = field(default=0, repr=False)

    @property
    def current_beat(self) -> Beat:
        return self.beats[min(self._current_index, len(self.beats) - 1)]

    @property
    def current_severity(self) -> int:
        return self.current_beat.severity

    def advance(self) -> None:
        if self._current_index < len(self.beats) - 1:
            self._current_index += 1

    def is_exhausted(self) -> bool:
        return self._current_index >= len(self.beats) - 1

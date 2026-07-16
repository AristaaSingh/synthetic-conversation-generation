# References
# [1] Bao, J., Wang, R., Wang, Y., Sun, A., Li, Y., Mi, F., & Xu, R. (2023).
#     A Synthetic Data Generation Framework for Grounded Dialogues.
#     In Proceedings of ACL 2023, pp. 10866–10882.
#
# [2] Morabito, R., Madhusudan, S., McDonald, T., & Emami, A. (2024).
#     STOP! Benchmarking Large Language Models with Sensitivity Testing on
#     Offensive Progressions. In Proceedings of EMNLP 2024, pp. 4221–4243.

# [3] Capodilupo, C. M., Nadal, K. L., Corman, L., Hamit, S., Lyons, O. B., &
#     Weinberg, A. (2010). The manifestation of gender microaggressions.
#     In D. W. Sue (Ed.), Microaggressions and Marginality, pp. 193–216. Wiley.

from dataclasses import dataclass, field
from typing import Optional


# Beat severity tiers are aligned with the STOP offensive progression scale [2].
@dataclass
class Beat:
    """
    One planned moment in a conversational arc.

    A beat is controlled on two independent axes:
      - `category` says WHAT KIND of microaggression is in play  (Capodilupo [3])
      - `severity` says HOW INTENSE it is                        (STOP [2])

    Previously only the severity axis existed; the *kind* of microaggression was
    left to the planner improvising inside `description`, loosely nudged by a
    single free-text world label. `category` makes that axis explicit and
    machine-checkable, which also means every generated conversation records
    which microaggression type was intended at which beat.

    topic       — what the characters are texting about in this beat
    category    — canonical microaggression category key, or None for a neutral
                  beat (severity 1). Must be a key in microaggression_taxonomy.
    severity    — STOP-scale severity tier [2]:
                    1 = neutral; 2 = subtle; 3 = noticeable;
                    4 = significant; 5 = acute
    exchanges   — how many back-and-forths (2 turns each) this beat needs.
                  Previously every beat was allotted a fixed 2 turns, which gave a
                  trivial logistical exchange exactly as much room as a relational
                  incident. Beat duration is now a planned property: a quick
                  hand-off needs 1 exchange, whereas a remark that lands badly needs
                  room to land, draw a reaction, and settle.
    description — the relational dynamic or behaviour pattern active here
    """
    topic: str
    severity: int
    description: str
    category: Optional[str] = None
    exchanges: int = 1

    @property
    def turns(self) -> int:
        """Turns this beat occupies (one exchange = one message from each character)."""
        return max(1, self.exchanges) * 2


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
    _turns_used_on_beat: int = field(default=0, repr=False)

    @property
    def current_beat(self) -> Optional[Beat]:
        """
        The beat being realised now, or None once the plan is spent.

        Returning None past the end is essential. The previous implementation
        clamped with `min(index, len-1)`, so it silently returned the LAST beat
        forever once exhausted. The pipeline tried to compensate by assigning
        `current_beat = None` at the bottom of the turn loop, but that was
        overwritten at the top of the next iteration — so the generator received
        the same final beat for every remaining turn (48 of 60 turns in a default
        run). Exhaustion is now owned by the flow itself and cannot be bypassed.
        """
        if self.is_exhausted():
            return None
        return self.beats[self._current_index]

    @property
    def current_severity(self) -> int:
        beat = self.current_beat
        return beat.severity if beat else 1

    def record_turn(self) -> None:
        """
        Register that one turn has been spent on the current beat, advancing to the
        next beat once this one has had the number of exchanges it asked for.

        Replaces the pipeline's fixed `session_turn_count % 2` advance, which gave
        every beat exactly 2 turns regardless of what it needed.
        """
        if self.is_exhausted():
            return
        self._turns_used_on_beat += 1
        if self._turns_used_on_beat >= self.beats[self._current_index].turns:
            self._current_index += 1
            self._turns_used_on_beat = 0

    def is_exhausted(self) -> bool:
        """True once every planned beat has had its full allotment of turns."""
        return self._current_index >= len(self.beats)

    def total_turns(self) -> int:
        """Total turns this plan covers — the session's planned length."""
        return sum(b.turns for b in self.beats)

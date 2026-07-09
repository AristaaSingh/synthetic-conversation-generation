# References
# [1] Liu, Y., Yao, J., Cheng, Y., An, Y., Chen, X., Feng, S., Huang, Y., Shen, S.,
#     Zhang, R., Du, K., & Jiang, J. (2025). LMCache: An Efficient KV Cache Layer
#     for Enterprise-Scale LLM Inference. arXiv:2510.09665.
#
# Design note: this implements the application-layer analogue of LMCache's
# cross-query KV reuse. LMCache persists GPU attention tensors across queries
# sharing a common prefix; here we persist explicit conversational commitments
# as structured entries so they survive rolling-summary compression.

from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class CommitmentEntry:
    """
    A single commitment or instruction that one character directed at another.

    speaker:    the character who issued the instruction
    recipient:  the character expected to honour it
    text:       the commitment in plain English (e.g. "stop sending follow-up messages")
    turn_index: conversation turn at which the commitment was issued
    """
    speaker: str
    recipient: str
    text: str
    turn_index: int


class CommitmentCache:
    """
    A key-value store of active commitments, keyed by recipient.

    Inspired by LMCache (Liu et al., 2025) which treats cached context as
    structured, addressable entries rather than opaque text blobs. Here the
    'key' is (recipient, normalised topic) and the 'value' is the commitment
    text. Unlike the rolling summary — which compresses many turns into prose
    and loses specificity — the commitment cache retains discrete instructions
    verbatim so they can be injected directly into the recipient's next prompt.

    Eviction: entries older than _TTL_TURNS turns are dropped to avoid
    surfacing stale obligations that no longer apply.
    """

    _TTL_TURNS: int = 40

    def __init__(self) -> None:
        self._entries: list[CommitmentEntry] = []

    def add(self, entry: CommitmentEntry) -> None:
        self._entries.append(entry)

    def get_for_recipient(self, recipient: str, current_turn: int) -> list[CommitmentEntry]:
        """Return all live commitments directed at recipient."""
        return [
            e for e in self._entries
            if e.recipient == recipient
            and (current_turn - e.turn_index) <= self._TTL_TURNS
        ]

    def evict_stale(self, current_turn: int) -> None:
        self._entries = [
            e for e in self._entries
            if (current_turn - e.turn_index) <= self._TTL_TURNS
        ]

    def to_dict_list(self) -> list[dict]:
        return [
            {
                "speaker": e.speaker,
                "recipient": e.recipient,
                "text": e.text,
                "turn_index": e.turn_index,
            }
            for e in self._entries
        ]

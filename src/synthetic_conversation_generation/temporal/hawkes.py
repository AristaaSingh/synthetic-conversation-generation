"""
Hawkes process simulation for inter-message timing in synthetic VAWG conversations.

Implements Ogata's (1988) thinning algorithm for a univariate Hawkes process
with an exponential decay kernel:

    λ(t) = μ + Σ_{t_i < t} α · exp(-β · (t - t_i))

Where:
    μ (mu)    — baseline rate: background message frequency (messages per minute)
    α (alpha) — excitation amplitude: how much each message raises the rate
    β (beta)  — decay rate: how fast the excitement dies down

Stability condition: α/β < 1 (the process must not explode).

Phase parameters are informed by:
    - Aoki et al. (2016): SMS-specific Hawkes parameters
    - Falkner et al. (2022): Hawkes parameters vary by relationship type
    - Hong et al. (2008): Heavy-tailed inter-message timing in SMS
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional
import numpy as np


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

@dataclass
class HawkesParameters:
    """
    Parameters for a Hawkes process with exponential kernel.

    All rates are in events per minute.
    """
    mu: float     # baseline rate
    alpha: float  # excitation amplitude
    beta: float   # decay rate

    def __post_init__(self):
        if self.mu <= 0:
            raise ValueError(f"mu must be positive, got {self.mu}")
        if self.alpha < 0 or self.beta <= 0:
            raise ValueError("alpha must be >= 0 and beta must be > 0")
        if self.alpha / self.beta >= 1:
            raise ValueError(
                f"Unstable Hawkes process: alpha/beta = {self.alpha/self.beta:.3f} must be < 1. "
                "Reduce alpha or increase beta."
            )


# Phase-specific parameters grounded in empirical SMS literature.
# Units: events per minute.
#
# early_contact  — low baseline, moderate excitement; small talk, testing the water
# escalation     — higher baseline and excitement; frequent, erratic exchanges
# post_incident  — very low baseline, low excitement; victim withdraws after incident
# re_initiation  — low baseline; perpetrator breaks silence after days
#
# These are starting values informed by Aoki et al. (2016) and Falkner et al. (2022).
# They can be tuned per scenario.
PHASE_PARAMETERS: dict[str, HawkesParameters] = {
    "early_contact": HawkesParameters(mu=0.005, alpha=0.3,  beta=0.5),
    "escalation":    HawkesParameters(mu=0.015, alpha=0.6,  beta=0.8),
    "post_incident": HawkesParameters(mu=0.001, alpha=0.15, beta=0.5),
    "re_initiation": HawkesParameters(mu=0.003, alpha=0.25, beta=0.6),
}


# ---------------------------------------------------------------------------
# Session boundary gaps
# ---------------------------------------------------------------------------

# Gap ranges in HOURS, sampled uniformly, applied when one session ends and the
# next begins.
#
# These are phase-dependent because the previous implementation used a fixed
# 4-24h gap regardless of what had just happened, which was both unrealistic and
# the main brake on the conversation's overall arc. The phases already model
# withdrawal at the within-session level (post_incident has a ~17h baseline
# precisely because someone has pulled back), yet a session ending immediately
# after a relational incident resumed within a day like any other. After a real
# incident, contact resumes in DAYS, not hours.
#
# Session boundaries are the dominant contributor to the conversation's total
# span: measurement of run 6543856 found only 2 gaps exceeding 4h across 60
# turns, and both were session boundaries. Making these phase-aware therefore
# lengthens the arc and improves realism at the same time.
SESSION_GAP_HOURS: dict[str, tuple[float, float]] = {
    # (minimum hours, additional uniform spread)
    "early_contact": (4, 20),    # later that day, or the next morning
    "escalation":    (2, 10),    # contact resumes quickly while things are live
    "post_incident": (16, 56),   # days of silence after a withdrawal (mean ~44h)
    "re_initiation": (8, 32),    # tentative re-contact after a break (mean ~24h)
}


# ---------------------------------------------------------------------------
# Core simulation — Ogata (1988) thinning algorithm
# ---------------------------------------------------------------------------

def simulate_hawkes(
    params: HawkesParameters,
    horizon_minutes: float,
    seed: Optional[int] = None,
) -> List[float]:
    """
    Simulate a Hawkes process over [0, horizon_minutes] using the thinning algorithm.

    The intensity λ(t) decreases monotonically between events and jumps up by alpha
    at each event, so the intensity at the start of each interval is a valid upper
    bound for the thinning step.

    Args:
        params:           Hawkes parameters (mu, alpha, beta).
        horizon_minutes:  Total time window to simulate over, in minutes.
        seed:             Optional random seed for reproducibility.

    Returns:
        Sorted list of event times in minutes from t=0.
    """
    rng = np.random.default_rng(seed)
    events: List[float] = []
    t = 0.0

    def intensity(s: float) -> float:
        lam = params.mu
        for t_i in events:
            lam += params.alpha * np.exp(-params.beta * (s - t_i))
        return lam

    while t < horizon_minutes:
        lam_upper = intensity(t)

        # Sample candidate inter-arrival from homogeneous Poisson at current rate
        w = rng.exponential(1.0 / lam_upper)
        t_candidate = t + w

        if t_candidate > horizon_minutes:
            break

        # Accept with probability λ(t_candidate) / λ_upper
        lam_candidate = intensity(t_candidate)
        if rng.uniform() <= lam_candidate / lam_upper:
            events.append(t_candidate)

        t = t_candidate

    return events


# ---------------------------------------------------------------------------
# Conversation timer — manages the running clock across a full conversation
# ---------------------------------------------------------------------------

@dataclass
class ConversationTimer:
    """
    Manages timestamp generation across a multi-turn conversation.

    Wraps the Hawkes simulation with phase-aware parameter switching and
    maintains a running clock anchored to a real start datetime. Each call
    to next_timestamp() advances the clock by the next Hawkes-sampled gap.

    The phase can be updated externally by the conversation state machine
    as the relationship dynamic evolves.
    """
    start_time: datetime
    phase: str = "early_contact"
    seed: Optional[int] = None

    # Internal state
    _current_time_minutes: float = field(default=0.0, init=False, repr=False)
    _event_history: List[float] = field(default_factory=list, init=False, repr=False)
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self):
        if self.phase not in PHASE_PARAMETERS:
            raise ValueError(
                f"Unknown phase '{self.phase}'. "
                f"Valid phases: {list(PHASE_PARAMETERS.keys())}"
            )
        self._rng = np.random.default_rng(self.seed)

    def set_phase(self, phase: str) -> None:
        """Switch to a different relationship phase. Takes effect on the next gap sample."""
        if phase not in PHASE_PARAMETERS:
            raise ValueError(
                f"Unknown phase '{phase}'. "
                f"Valid phases: {list(PHASE_PARAMETERS.keys())}"
            )
        self.phase = phase

    def _sample_gap(self) -> float:
        """
        Sample the next inter-message gap in minutes using the thinning algorithm.

        Uses the current phase parameters and the full event history to compute
        the current intensity, then samples one event from the process.
        """
        params = PHASE_PARAMETERS[self.phase]
        t = self._current_time_minutes

        def intensity(s: float) -> float:
            lam = params.mu
            for t_i in self._event_history:
                lam += params.alpha * np.exp(-params.beta * (s - t_i))
            return lam

        # Thinning: sample candidate gaps until one is accepted
        while True:
            lam_upper = intensity(t)
            w = self._rng.exponential(1.0 / lam_upper)
            t_candidate = t + w

            lam_candidate = intensity(t_candidate)
            if self._rng.uniform() <= lam_candidate / lam_upper:
                return t_candidate - self._current_time_minutes

            t = t_candidate

    def next_timestamp(self) -> tuple[datetime, float]:
        """
        Advance the conversation clock by one message gap.

        Returns:
            (timestamp, gap_minutes) — the datetime of the next message
            and the gap in minutes since the previous message.
        """
        gap_minutes = self._sample_gap()
        self._current_time_minutes += gap_minutes
        self._event_history.append(self._current_time_minutes)

        timestamp = self.start_time + timedelta(minutes=self._current_time_minutes)
        return timestamp, gap_minutes

    def force_gap_hours(
        self,
        between: Optional[float] = None,
        spread: Optional[float] = None,
    ) -> float:
        """
        Jump the clock forward by a random gap to simulate a session boundary.

        Used when one conversation session has ended and a new one is about to
        begin. The gap is sampled uniformly from [between, between + spread]
        hours. The event history is cleared so the next session's Hawkes
        excitation starts fresh rather than continuing the previous burst — this
        resets only the short-term burst dynamics, not the relationship's phase
        or any narrative state.

        If `between`/`spread` are not given, they are drawn from
        SESSION_GAP_HOURS for the CURRENT PHASE, so that (for example) a session
        ending in post_incident resumes days later rather than the same evening.

        Args:
            between: Minimum gap in hours. Defaults to the current phase's value.
            spread:  Additional random hours on top. Defaults to the phase's value.

        Returns:
            The gap applied, in hours.
        """
        if between is None or spread is None:
            phase_between, phase_spread = SESSION_GAP_HOURS[self.phase]
            between = phase_between if between is None else between
            spread = phase_spread if spread is None else spread

        gap_hours = between + self._rng.uniform(0, spread)
        self._current_time_minutes += gap_hours * 60
        self._event_history = []  # fresh start for next session's excitation
        return gap_hours

    @property
    def current_time(self) -> datetime:
        """Current datetime of the conversation clock."""
        return self.start_time + timedelta(minutes=self._current_time_minutes)

    @property
    def elapsed_days(self) -> float:
        """Total days elapsed since the start of the conversation."""
        return self._current_time_minutes / 1440.0

    @property
    def elapsed_minutes(self) -> float:
        """Total minutes elapsed since the start of the conversation."""
        return self._current_time_minutes

"""
A fake ModelProvider that returns canned responses instead of calling an LLM.

Why this exists
---------------
The pipeline makes 400-900 LLM calls per run, so exercising its control flow
against a real model is slow, needs a GPU, and is non-deterministic. This fake
satisfies every query's response schema with canned data, which means the turn
loop, beat advancement, session boundaries, checkpointing and the Hawkes timing
can all be tested **in milliseconds, on a laptop, with no GPU and no Ollama**.

This is not only a test utility. It found a real bug: the beat-exhaustion defect
(project_record.md 23.4) was invisible in production logs but obvious the moment
the beat handed to each turn could be recorded directly.

The provider dispatches on the *response schema* rather than the query class,
so it keeps working when queries are added or renamed as long as their schemas
stay distinct.

Usage
-----
    from synthetic_conversation_generation.testing.fake_provider import FakeModelProvider

    provider = FakeModelProvider()
    run_pipeline(model_provider=provider, model_id="fake", ...)
    print(provider.call_counts)   # {'message': 40, 'consistency': 40, ...}
"""
from __future__ import annotations

from typing import Callable, Dict, Optional

from synthetic_conversation_generation.llm_queries.llm_query import ModelProvider


class FakeModelProvider(ModelProvider):
    """
    Canned-response provider for testing pipeline control flow.

    Args:
        beats:        Beat dicts each dialogue-flow plan should return. Defaults to a
                      3-beat plan with varied `exchanges` so beat-length handling is
                      exercised.
        phase_at:     Callable mapping the assessment number (1-based) to a phase
                      name, for simulating a phase trajectory. Defaults to a constant
                      "early_contact".
        session_end_at: Assessment number at which `session_ended` returns True, to
                      test the assessor-driven session boundary. None = never.
        tension_at:   Callable mapping assessment number to a tension level.
        fail_after:   Raise KeyboardInterrupt after this many calls, simulating a
                      SLURM timeout mid-run. None = never.
        consistent:   What the persona filter returns. False exercises the retry loop.
    """

    def __init__(
        self,
        beats: Optional[list[dict]] = None,
        phase_at: Optional[Callable[[int], str]] = None,
        session_end_at: Optional[int] = None,
        tension_at: Optional[Callable[[int], int]] = None,
        fail_after: Optional[int] = None,
        consistent: bool = True,
    ):
        self.beats = beats if beats is not None else [
            {"topic": "standup timing", "category": "none",
             "severity": 1, "exchanges": 1, "description": "quick logistical hand-off"},
            {"topic": "code review feedback", "category": "assumptions_of_inferiority",
             "severity": 2, "exchanges": 2, "description": "he re-explains her own fix to her"},
            {"topic": "deployment plan", "category": "second_class_citizenship",
             "severity": 3, "exchanges": 2, "description": "he decides without asking her"},
        ]
        self.phase_at = phase_at or (lambda n: "early_contact")
        self.tension_at = tension_at or (lambda n: 2)
        self.session_end_at = session_end_at
        self.fail_after = fail_after
        self.consistent = consistent

        self.calls = 0
        self.assessments = 0
        self.call_counts: Dict[str, int] = {}
        self.prompts: Dict[str, list[str]] = {}

    # -- ModelProvider interface ------------------------------------------------

    def response_format(self, response_schema: Dict) -> Dict:
        return response_schema

    def query(self, user_msg: str, response_schema: Dict, model_id: str, timeout: int = 60):
        self.calls += 1
        if self.fail_after is not None and self.calls > self.fail_after:
            raise KeyboardInterrupt("simulated SLURM timeout")

        props = response_schema.get("properties", {})
        kind = self._classify(props)
        self.call_counts[kind] = self.call_counts.get(kind, 0) + 1
        self.prompts.setdefault(kind, []).append(user_msg)

        return getattr(self, f"_{kind}")()

    # -- dispatch ---------------------------------------------------------------

    @staticmethod
    def _classify(props: Dict) -> str:
        """Identify a query by its response schema, not its class name."""
        if "beats" in props:
            return "dialogue_flow"
        if "text" in props:
            return "message"
        if "is_consistent" in props:
            return "consistency"
        if "phase" in props:
            return "state_assessment"
        if "commitments" in props:
            return "commitment"
        if "events" in props:
            return "rolling_summary"
        return "unknown"

    # -- canned responses -------------------------------------------------------

    def _dialogue_flow(self):
        return {"beats": [dict(b) for b in self.beats]}

    def _message(self):
        return {"text": f"message {self.calls}"}

    def _consistency(self):
        return {"is_consistent": self.consistent, "reason": "canned"}

    def _state_assessment(self):
        self.assessments += 1
        n = self.assessments
        return {
            "phase": self.phase_at(n),
            "summary": f"canned state at assessment {n}",
            "tension_level": self.tension_at(n),
            "incident_occurred": self.phase_at(n) == "post_incident",
            "detected_categories": ["assumptions_of_inferiority"],
            "session_ended": self.session_end_at is not None and n == self.session_end_at,
        }

    def _commitment(self):
        return {"commitments": []}

    def _rolling_summary(self):
        return {"events": "e", "details": "d", "open_threads": "", "dynamic": "y"}

    def _unknown(self):
        return {}

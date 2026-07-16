"""
Control-flow tests for the generation pipeline.

Run with: python -m synthetic_conversation_generation.test_pipeline

No LLM, no GPU, no Ollama — every test runs against FakeModelProvider in
milliseconds. Each test corresponds to a defect that was actually found and
fixed during the code assessment (project_record.md 23), and exists to stop it
recurring silently.
"""
from __future__ import annotations

import json
import logging
import tempfile
from datetime import datetime
from pathlib import Path

from synthetic_conversation_generation.data_models.character_card import CharacterCard
from synthetic_conversation_generation.data_models.world import World
from synthetic_conversation_generation.testing.fake_provider import FakeModelProvider
from synthetic_conversation_generation import pipeline as pl

logging.disable(logging.INFO)

_ROOT = Path(__file__).resolve().parents[2]
CHAR_A = _ROOT / "data/characters/victims/sophie_walker.yaml"
CHAR_B = _ROOT / "data/characters/perpetrators/ryan_chambers.yaml"
WORLD = _ROOT / "data/worlds/uk_tech_company.yaml"


def _inputs():
    return (CharacterCard.from_yaml(str(CHAR_A)),
            CharacterCard.from_yaml(str(CHAR_B)),
            World.from_yaml(str(WORLD)))


def _run(provider, out=None, **kw):
    a, b, w = _inputs()
    kw.setdefault("max_turns", 30)
    kw.setdefault("target_days", 999)   # effectively disabled unless a test sets it
    kw.setdefault("hawkes_seed", 1)
    return pl.run_pipeline(
        model_provider=provider, model_id="fake",
        character_a=a, character_b=b, world=w,
        conversation_id="test", output_path=out, **kw,
    )


def _record_beats():
    """Patch CharacterMessageQuery to record the beat handed to each turn."""
    import synthetic_conversation_generation.llm_queries.character_message_query as cmq
    seen = []
    original = cmq.CharacterMessageQuery.__init__

    def spy(self, *a, **k):
        seen.append(k.get("current_beat"))
        original(self, *a, **k)

    cmq.CharacterMessageQuery.__init__ = spy
    return seen, lambda: setattr(cmq.CharacterMessageQuery, "__init__", original)


# ---------------------------------------------------------------------------
# 23.4 — beat exhaustion
# ---------------------------------------------------------------------------

def test_dialogue_flow_returns_none_when_exhausted():
    """
    The core invariant behind 23.4, tested directly on the dataclass.

    This must be a unit test, not an integration test. Two independent
    mechanisms now prevent a stale beat reaching the generator: `current_beat`
    returns None past the end (this test), and beat exhaustion ends the session
    (23.7). In a full pipeline run the second masks the first, so an integration
    test would still pass if this invariant were broken — as was confirmed by
    reintroducing the original clamped `min(index, len-1)` implementation and
    watching the integration test pass regardless. Only a direct test of the
    contract catches a regression here.
    """
    from synthetic_conversation_generation.data_models.dialogue_flow import Beat, DialogueFlow

    flow = DialogueFlow(session_number=1, beats=[
        Beat(topic="one", severity=1, description="d", category=None, exchanges=1),
        Beat(topic="two", severity=2, description="d",
             category="assumptions_of_inferiority", exchanges=1),
    ])

    assert flow.current_beat.topic == "one"
    flow.record_turn(); flow.record_turn()          # spend beat one (1 exchange = 2 turns)
    assert flow.current_beat.topic == "two", "should advance to the second beat"
    assert not flow.is_exhausted()

    flow.record_turn(); flow.record_turn()          # spend beat two
    assert flow.is_exhausted(), "plan should be spent"
    assert flow.current_beat is None, (
        "current_beat MUST return None once exhausted — returning the last beat "
        "is the original 23.4 defect, which silently fed one beat to 80% of a run"
    )

    # Must stay None however many further turns are recorded.
    for _ in range(10):
        flow.record_turn()
        assert flow.current_beat is None

    assert flow.total_turns() == 4
    print("[PASS] DialogueFlow contract — current_beat is None past the end, permanently")


def test_no_beat_outlives_its_budget():
    """
    Integration counterpart to the exhaustion bug (23.4).

    The original defect handed ONE beat to the generator for 48 consecutive turns.
    The invariant: no beat may occupy more consecutive turns than `exchanges * 2`.
    (A "None" turn is unreachable in practice — plans are always an even number of
    turns and exhaustion ends the session — so the None path is covered by the unit
    test above rather than here.)
    """
    seen, restore = _record_beats()
    try:
        _run(FakeModelProvider(), max_turns=40)
    finally:
        restore()

    longest, current, prev = 0, 0, object()
    worst = None
    for b in seen:
        key = (b.topic, b.exchanges) if b else None
        if key == prev:
            current += 1
        else:
            current, prev = 1, key
        if b and current > longest:
            longest, worst = current, b

    assert worst is not None, "no beats were handed to the generator"
    assert longest <= worst.turns, (
        f"beat '{worst.topic}' ({worst.exchanges}ex = {worst.turns} turns) was handed "
        f"to the generator for {longest} consecutive turns — stale beat reuse"
    )
    print(f"[PASS] no stale beat reuse — longest run {longest} turns "
          f"(budget {worst.turns}) over {len(seen)} turns")


def test_beat_lengths_are_respected():
    """23.5 — a beat must occupy exactly `exchanges * 2` turns, not a fixed 2."""
    seen, restore = _record_beats()
    try:
        _run(FakeModelProvider(beats=[
            {"topic": "quick", "category": "none", "severity": 1,
             "exchanges": 1, "description": "d"},
            {"topic": "long", "category": "assumptions_of_inferiority", "severity": 3,
             "exchanges": 3, "description": "d"},
        ]), max_turns=8)   # exactly one plan: (1+3) exchanges = 8 turns
    finally:
        restore()

    counts: dict[str, int] = {}
    for b in seen:
        if b:
            counts[b.topic] = counts.get(b.topic, 0) + 1
    assert counts.get("quick") == 2, f"1-exchange beat should get 2 turns, got {counts}"
    assert counts.get("long") == 6, f"3-exchange beat should get 6 turns, got {counts}"
    print(f"[PASS] variable beat length — {counts}")


def test_beat_description_is_hidden_from_the_victim():
    """
    The beat must be perspective-aware (24.2).

    `Beat.description` states the PERPETRATOR's behaviour. Showing it to the victim
    hands her his script: in run 6641761 Sophie agreed to organise catering one turn
    BEFORE Ryan asked her to, because her prompt contained "Ryan says to Sophie,
    'Can you set up the catering?'". Both characters may see the topic; only the
    perpetrator may see the intent.
    """
    from synthetic_conversation_generation.llm_queries.character_message_query import (
        CharacterMessageQuery,
    )
    from synthetic_conversation_generation.data_models.conversation import Conversation
    from synthetic_conversation_generation.data_models.dialogue_flow import Beat

    a, b, w = _inputs()
    beat = Beat(topic="team lunch planning", severity=4,
                description="Ryan says 'Can you set up the catering?', expecting her to handle it",
                category="traditional_gender_roles", exchanges=1)
    convo = Conversation(id="t", user_id=a.name, messages=[])

    def prompt_for(sender, receiver, is_a):
        return CharacterMessageQuery(
            model_provider=None, model_id="x", conversation=convo,
            sender=sender, receiver=receiver, world=w,
            is_sender_character_a=is_a, current_beat=beat,
        ).generate_prompt()

    victim = prompt_for(a, b, True)      # Sophie = character_a
    perp = prompt_for(b, a, False)       # Ryan   = character_b

    assert "team lunch planning" in victim, "victim should still receive the topic"
    assert "Can you set up the catering" not in victim, (
        "victim's prompt contains the perpetrator's planned line — she will "
        "pre-empt it (the 6641761 catering defect)"
    )
    assert "Can you set up the catering" in perp, "perpetrator must receive the intent"
    print("[PASS] beat perspective — victim gets the topic only, perpetrator gets the intent")


# ---------------------------------------------------------------------------
# 23.7 — session ends
# ---------------------------------------------------------------------------

def test_assessor_can_end_session_early():
    """The assessor's `session_ended` must trigger a boundary before beats run out."""
    p = FakeModelProvider(session_end_at=1)
    _, _, _, flows, _ = _run(p, max_turns=30)
    assert len(flows) > 1, "an early session end should trigger a replan"
    print(f"[PASS] assessor-driven session end — {len(flows)} sessions planned")


def test_exhausted_plan_ends_session():
    """Beat exhaustion is the backstop trigger when no natural ending arrives."""
    p = FakeModelProvider(session_end_at=None)   # assessor never ends it
    _, _, _, flows, _ = _run(p, max_turns=30)
    assert len(flows) > 1, "exhausting the plan should still end the session"
    print(f"[PASS] exhaustion-driven session end — {len(flows)} sessions planned")


def test_completion_query_is_gone():
    """23.7 — the deleted ConversationCompletionQuery must not be called."""
    p = FakeModelProvider()
    _run(p, max_turns=20)
    assert "unknown" not in p.call_counts, (
        f"an unrecognised query was issued: {p.call_counts}"
    )
    assert not hasattr(pl, "ConversationCompletionQuery"), "completion query still imported"
    print(f"[PASS] no completion query — calls: {p.call_counts}")


# ---------------------------------------------------------------------------
# 23.6 — budget coherence
# ---------------------------------------------------------------------------

def test_planner_receives_the_exchange_budget():
    """
    23.6 — the beat count is not a fixed number; the planner is given a budget and
    chooses. The budget is now a direct parameter rather than being derived from
    max_sessions (which coupled two unrelated things: raising the session ceiling
    silently shortened every session).
    """
    p = FakeModelProvider()
    _run(p, max_turns=20, exchange_budget=5)
    assert "budget of about 5 exchanges" in p.prompts["dialogue_flow"][0], \
        "planner should receive the exchange budget verbatim"
    print("[PASS] planner receives its exchange budget")


# ---------------------------------------------------------------------------
# 23.3 — checkpointing
# ---------------------------------------------------------------------------

def test_checkpoint_survives_mid_run_kill():
    """
    23.3 — a SLURM timeout must leave a valid, parseable file. Writes are atomic
    (tmp + os.replace), so a kill mid-write cannot truncate the output or destroy
    the previous checkpoint.
    """
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "run.json"
        try:
            _run(FakeModelProvider(fail_after=20), out=out, max_turns=60)
        except KeyboardInterrupt:
            pass

        assert out.exists(), "no checkpoint was written before the kill"
        data = json.loads(out.read_text())          # raises if truncated
        assert data["complete"] is False, "a killed run must not be marked complete"
        assert len(data["messages"]) > 0
        assert not out.with_suffix(".json.tmp").exists(), "temp file left behind"
        print(f"[PASS] checkpoint survives kill — {data['turns_generated']} turns, "
              f"complete={data['complete']}")


def test_completed_run_is_marked_complete():
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "run.json"
        _run(FakeModelProvider(), out=out, max_turns=12)
        data = json.loads(out.read_text())
        assert data["complete"] is True
        assert data["turns_generated"] == 12
        print(f"[PASS] completed run marked complete — {data['turns_generated']} turns")


# ---------------------------------------------------------------------------
# 23.1 / 23.2 — taxonomy
# ---------------------------------------------------------------------------

def test_output_is_labelled_with_taxonomy():
    """23.1/23.2 — every run must be self-labelled in the canonical vocabulary."""
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "run.json"
        _run(FakeModelProvider(), out=out, max_turns=12)
        data = json.loads(out.read_text())

        assert data["vawg_categories"], "world palette missing from output"
        beat = data["dialogue_flows"][0]["beats"][1]
        assert beat["category"] == "assumptions_of_inferiority"
        assert beat["exchanges"] == 2
        assert data["final_state"]["detected_categories"], "assessor categories missing"
        print(f"[PASS] taxonomy-labelled output — palette={len(data['vawg_categories'])} "
              f"categories, beats carry category+exchanges")


def test_world_rejects_unknown_category():
    """23.1 — a stale EXIST label must fail loudly, not be silently prompted."""
    for bad in (["STEREOTYPING-DOMINANCE"], []):
        try:
            World(title="Bad", setting="", relationship="", vawg_categories=bad,
                  character_a_role="", character_b_role="")
        except ValueError:
            continue
        raise AssertionError(f"World accepted invalid categories: {bad}")
    print("[PASS] world validates categories — unknown and empty both rejected")


# ---------------------------------------------------------------------------
# 23.8 — phase-aware session gaps
# ---------------------------------------------------------------------------

def test_post_incident_sessions_resume_days_later():
    """
    23.8 — session gaps must depend on the phase the session ended in. A gap after
    a withdrawal should be days; a gap during escalation should be hours.
    """
    from synthetic_conversation_generation.temporal.hawkes import ConversationTimer

    gaps = {}
    for phase in ("escalation", "post_incident"):
        t = ConversationTimer(start_time=datetime(2026, 1, 5, 9, 0), phase=phase, seed=3)
        gaps[phase] = [t.force_gap_hours() for _ in range(200)]

    mean_esc = sum(gaps["escalation"]) / 200
    mean_post = sum(gaps["post_incident"]) / 200
    assert mean_post > mean_esc * 3, (
        f"post_incident gaps ({mean_post:.0f}h) should dwarf escalation ({mean_esc:.0f}h)"
    )
    assert min(gaps["post_incident"]) >= 16, "post_incident gap should never be same-day"
    print(f"[PASS] phase-aware gaps — escalation ~{mean_esc:.0f}h, "
          f"post_incident ~{mean_post:.0f}h")


def test_run_terminates_on_target_duration():
    """
    The conversation must end because the SIMULATED CLOCK reached target_days —
    not because a turn or session ceiling cut it off. Turn count is emergent.
    """
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "run.json"
        # a plausible trajectory: early contact -> escalation -> incident
        _run(FakeModelProvider(
            phase_at=lambda n: "early_contact" if n < 20 else
                               ("escalation" if n < 50 else "post_incident"),
            tension_at=lambda n: min(5, 1 + n // 12),
        ), out=out, target_days=14, max_turns=300)

        data = json.loads(out.read_text())
        ts = [datetime.strptime(m["timestamp"], "%Y-%m-%d %H:%M") for m in data["messages"]]
        span_days = (ts[-1] - ts[0]).total_seconds() / 86400
        gaps = sorted((ts[i + 1] - ts[i]).total_seconds() / 60 for i in range(len(ts) - 1))
        median = gaps[len(gaps) // 2]

        assert span_days >= 14, (
            f"run must continue until target_days is reached, got {span_days:.1f}d"
        )
        assert data["turns_generated"] < 300, (
            "the circuit breaker fired — duration should have terminated the run"
        )
        # Bursts must survive: raising the median would mean we broke the Hawkes model.
        assert median < 60, f"median gap {median:.0f}m — bursts destroyed"
        print(f"[PASS] duration termination — {span_days:.1f}d "
              f"(target 14), {data['turns_generated']} turns emergent, "
              f"median gap {median:.1f}m (bursts intact)")


TESTS = [
    test_dialogue_flow_returns_none_when_exhausted,
    test_no_beat_outlives_its_budget,
    test_beat_lengths_are_respected,
    test_beat_description_is_hidden_from_the_victim,
    test_assessor_can_end_session_early,
    test_exhausted_plan_ends_session,
    test_completion_query_is_gone,
    test_planner_receives_the_exchange_budget,
    test_checkpoint_survives_mid_run_kill,
    test_completed_run_is_marked_complete,
    test_output_is_labelled_with_taxonomy,
    test_world_rejects_unknown_category,
    test_post_incident_sessions_resume_days_later,
    test_run_terminates_on_target_duration,
]


if __name__ == "__main__":
    print("=" * 72)
    print("PIPELINE CONTROL-FLOW TESTS (no LLM, no GPU)")
    print("=" * 72)
    failed = 0
    for t in TESTS:
        try:
            t()
        except AssertionError as e:
            failed += 1
            print(f"[FAIL] {t.__name__}: {e}")
        except Exception as e:
            failed += 1
            print(f"[ERROR] {t.__name__}: {type(e).__name__}: {e}")
    print("=" * 72)
    print(f"{len(TESTS) - failed}/{len(TESTS)} passed")
    raise SystemExit(1 if failed else 0)

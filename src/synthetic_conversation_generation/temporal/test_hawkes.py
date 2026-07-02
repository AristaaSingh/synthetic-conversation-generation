"""
Quick sanity checks for the Hawkes process implementation.
Run with: python -m synthetic_conversation_generation.temporal.test_hawkes
"""

from datetime import datetime
from .hawkes import simulate_hawkes, ConversationTimer, PHASE_PARAMETERS


def test_simulate_produces_events():
    events = simulate_hawkes(PHASE_PARAMETERS["escalation"], horizon_minutes=1440, seed=42)
    assert len(events) > 0, "Expected events in a 24-hour escalation window"
    assert all(0 < e <= 1440 for e in events), "All events must be within horizon"
    assert events == sorted(events), "Events must be in ascending order"
    print(f"[PASS] simulate_hawkes — {len(events)} events over 24h (escalation phase)")


def test_phase_ordering():
    """Escalation phase should produce more events than post_incident phase."""
    n_escalation = len(simulate_hawkes(PHASE_PARAMETERS["escalation"], 1440, seed=0))
    n_post       = len(simulate_hawkes(PHASE_PARAMETERS["post_incident"], 1440, seed=0))
    assert n_escalation > n_post, (
        f"Expected escalation ({n_escalation}) > post_incident ({n_post})"
    )
    print(f"[PASS] phase ordering — escalation: {n_escalation} events, post_incident: {n_post} events")


def test_stability_check():
    from .hawkes import HawkesParameters
    try:
        HawkesParameters(mu=0.01, alpha=1.0, beta=0.5)  # alpha/beta = 2.0, unstable
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    print("[PASS] stability check — unstable parameters correctly rejected")


def test_conversation_timer():
    start = datetime(2024, 1, 1, 9, 0, 0)
    timer = ConversationTimer(start_time=start, phase="early_contact", seed=42)

    timestamps = []
    for _ in range(10):
        ts, gap = timer.next_timestamp()
        timestamps.append((ts, gap))
        assert ts > start, "Timestamp must be after start"
        assert gap > 0, "Gap must be positive"

    print(f"[PASS] ConversationTimer — 10 timestamps generated")
    print(f"       First gap:  {timestamps[0][1]:.1f} min  ({timestamps[0][0].strftime('%Y-%m-%d %H:%M')})")
    print(f"       Last gap:   {timestamps[-1][1]:.1f} min ({timestamps[-1][0].strftime('%Y-%m-%d %H:%M')})")
    print(f"       Total elapsed: {timer.elapsed_days:.2f} days")


def test_phase_switch():
    start = datetime(2024, 1, 1, 9, 0, 0)
    timer = ConversationTimer(start_time=start, phase="early_contact", seed=1)

    for _ in range(5):
        timer.next_timestamp()

    timer.set_phase("post_incident")
    _, gap_after = timer.next_timestamp()

    print(f"[PASS] phase switch — gap after switching to post_incident: {gap_after:.1f} min")


def test_fourteen_day_simulation():
    """Simulate a full 14-day conversation arc with phase transitions."""
    start = datetime(2024, 1, 1, 9, 0, 0)
    timer = ConversationTimer(start_time=start, phase="early_contact", seed=7)

    phase_schedule = [
        (3,  "escalation"),
        (10, "post_incident"),
        (12, "re_initiation"),
    ]
    phase_idx = 0
    messages = []

    while timer.elapsed_days < 14:
        # Switch phase when scheduled
        if phase_idx < len(phase_schedule):
            day_threshold, next_phase = phase_schedule[phase_idx]
            if timer.elapsed_days >= day_threshold:
                timer.set_phase(next_phase)
                phase_idx += 1

        ts, gap = timer.next_timestamp()
        messages.append((ts, timer.phase, gap))

    print(f"[PASS] 14-day simulation — {len(messages)} messages generated")

    # Print a sample across the arc
    indices = [0, len(messages)//4, len(messages)//2, 3*len(messages)//4, -1]
    print("       Sample messages across arc:")
    for i in indices:
        ts, phase, gap = messages[i]
        print(f"         [{ts.strftime('%Y-%m-%d %H:%M')}] phase={phase:<15} gap={gap:>8.1f} min")


if __name__ == "__main__":
    test_simulate_produces_events()
    test_phase_ordering()
    test_stability_check()
    test_conversation_timer()
    test_phase_switch()
    test_fourteen_day_simulation()
    print("\nAll tests passed.")

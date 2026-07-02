"""
Statistical validation of the Hawkes process implementation.

Checks:
1. Inter-event gap distribution is heavy-tailed (as per Hong et al. 2008, Aoki et al. 2016)
2. The process is self-exciting — events cluster in time (burstiness check)
3. Phase parameters produce measurably different gap distributions
4. 14-day arc produces a plausible message count and timing shape

Run with: python -m synthetic_conversation_generation.temporal.validate_hawkes
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from datetime import datetime

from .hawkes import simulate_hawkes, ConversationTimer, PHASE_PARAMETERS


def compute_gaps(events: list[float]) -> np.ndarray:
    """Convert absolute event times to inter-event gaps."""
    arr = np.array(events)
    return np.diff(arr) if len(arr) > 1 else np.array([])


# ---------------------------------------------------------------------------
# 1. Heavy-tail check — log-log plot of gap distribution
# ---------------------------------------------------------------------------

def plot_gap_distribution(ax, events, phase_name, color):
    gaps = compute_gaps(events)
    if len(gaps) < 2:
        return

    # Log-log histogram — a straight line here = power law (heavy tail)
    bins = np.logspace(np.log10(gaps.min() + 1e-6), np.log10(gaps.max()), 30)
    counts, edges = np.histogram(gaps, bins=bins)
    centres = (edges[:-1] + edges[1:]) / 2
    mask = counts > 0

    ax.loglog(centres[mask], counts[mask], 'o-', color=color,
              label=phase_name, markersize=4, linewidth=1.5)

    # Fit a log-normal (Stouffer et al. 2006) and overlay
    mu_ln, std_ln = np.log(gaps).mean(), np.log(gaps).std()
    x = np.logspace(np.log10(gaps.min() + 1e-6), np.log10(gaps.max()), 200)
    pdf = stats.lognorm.pdf(x, s=std_ln, scale=np.exp(mu_ln))
    # Scale to match histogram counts
    ax.loglog(x, pdf * len(gaps) * (edges[1] / edges[0] - 1) * x, '--',
              color=color, alpha=0.5, linewidth=1)


# ---------------------------------------------------------------------------
# 2. Burstiness check — coefficient of variation
#    CV > 1 means bursty (heavy-tailed); CV = 1 means Poisson; CV < 1 means regular
# ---------------------------------------------------------------------------

def burstiness(gaps: np.ndarray) -> float:
    if len(gaps) < 2:
        return 0.0
    return gaps.std() / gaps.mean()


# ---------------------------------------------------------------------------
# 3. 14-day arc — messages per day per phase
# ---------------------------------------------------------------------------

def simulate_arc(seed=42):
    start = datetime(2024, 1, 1, 9, 0, 0)
    timer = ConversationTimer(start_time=start, phase="early_contact", seed=seed)

    phase_schedule = [
        (3,  "escalation"),
        (10, "post_incident"),
        (12, "re_initiation"),
    ]
    phase_idx = 0
    messages = []  # (day_number, gap_minutes, phase)

    while timer.elapsed_days < 14:
        if phase_idx < len(phase_schedule):
            day_threshold, next_phase = phase_schedule[phase_idx]
            if timer.elapsed_days >= day_threshold:
                timer.set_phase(next_phase)
                phase_idx += 1

        ts, gap = timer.next_timestamp()
        messages.append((int(timer.elapsed_days), gap, timer.phase))

    return messages


# ---------------------------------------------------------------------------
# Main — generate plots and print statistics
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("HAWKES PROCESS VALIDATION")
    print("=" * 60)

    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    ax_dist   = fig.add_subplot(gs[0, 0])   # gap distributions per phase
    ax_burst  = fig.add_subplot(gs[0, 1])   # burstiness per phase
    ax_arc    = fig.add_subplot(gs[1, 0])   # messages per day across 14-day arc
    ax_gaps14 = fig.add_subplot(gs[1, 1])   # gap sizes across the arc

    colours = {
        "early_contact":  "#4C72B0",
        "escalation":     "#DD8452",
        "post_incident":  "#55A868",
        "re_initiation":  "#C44E52",
    }

    # --- Plot 1 & 2: gap distribution and burstiness per phase ---
    burstiness_scores = {}
    for phase, params in PHASE_PARAMETERS.items():
        # Simulate 30 days to get enough events for statistics
        events = simulate_hawkes(params, horizon_minutes=30 * 1440, seed=0)
        gaps = compute_gaps(events)

        print(f"\nPhase: {phase}")
        print(f"  Events in 30 days : {len(events)}")
        if len(gaps) > 0:
            print(f"  Median gap        : {np.median(gaps):.1f} min  "
                  f"({np.median(gaps)/60:.1f} h)")
            print(f"  Mean gap          : {np.mean(gaps):.1f} min")
            print(f"  Min gap           : {np.min(gaps):.2f} min")
            print(f"  Max gap           : {np.max(gaps):.1f} min  "
                  f"({np.max(gaps)/1440:.1f} days)")
            cv = burstiness(gaps)
            burstiness_scores[phase] = cv
            print(f"  Burstiness (CV)   : {cv:.3f}  "
                  f"({'bursty ✓' if cv > 1 else 'Poisson-like' if cv > 0.8 else 'regular'})")

            # Log-normal fit
            mu_ln = np.log(gaps).mean()
            std_ln = np.log(gaps).std()
            print(f"  Log-normal fit    : μ={mu_ln:.2f}, σ={std_ln:.2f}")

        plot_gap_distribution(ax_dist, events, phase, colours[phase])

    ax_dist.set_xlabel("Gap (minutes)", fontsize=11)
    ax_dist.set_ylabel("Count", fontsize=11)
    ax_dist.set_title("Inter-message gap distribution\n(log-log, dashed = log-normal fit)", fontsize=11)
    ax_dist.legend(fontsize=9)

    # Burstiness bar chart
    phases = list(burstiness_scores.keys())
    cvs = [burstiness_scores[p] for p in phases]
    bars = ax_burst.bar(range(len(phases)), cvs,
                        color=[colours[p] for p in phases], alpha=0.85)
    ax_burst.axhline(1.0, color='black', linestyle='--', linewidth=1,
                     label='CV=1 (Poisson)')
    ax_burst.set_xticks(range(len(phases)))
    ax_burst.set_xticklabels([p.replace("_", "\n") for p in phases], fontsize=9)
    ax_burst.set_ylabel("Coefficient of Variation (σ/μ)", fontsize=11)
    ax_burst.set_title("Burstiness per phase\n(CV > 1 = heavy-tailed, not Poisson)", fontsize=11)
    ax_burst.legend(fontsize=9)

    # --- Plot 3 & 4: 14-day arc ---
    arc_messages = simulate_arc(seed=42)

    # Messages per day
    messages_per_day = np.zeros(14)
    for day, gap, phase in arc_messages:
        if day < 14:
            messages_per_day[day] += 1

    phase_colours_per_day = []
    day_phases = {}
    for day, gap, phase in arc_messages:
        if day < 14:
            day_phases[day] = phase
    for d in range(14):
        phase_colours_per_day.append(colours.get(day_phases.get(d, "early_contact"), "#999"))

    ax_arc.bar(range(14), messages_per_day, color=phase_colours_per_day, alpha=0.85)
    ax_arc.set_xlabel("Day", fontsize=11)
    ax_arc.set_ylabel("Messages", fontsize=11)
    ax_arc.set_title("Messages per day — 14-day arc\n(colour = phase)", fontsize=11)

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=p) for p, c in colours.items()]
    ax_arc.legend(handles=legend_elements, fontsize=8)

    # Gap sizes across arc
    all_gaps = [gap for _, gap, _ in arc_messages]
    all_days = [day + (arc_messages[i][1] / 1440)
                for i, (day, _, _) in enumerate(arc_messages)]
    ax_gaps14.scatter(all_days[:len(all_gaps)], all_gaps,
                      alpha=0.3, s=8, color="#4C72B0")
    ax_gaps14.set_yscale("log")
    ax_gaps14.set_xlabel("Day", fontsize=11)
    ax_gaps14.set_ylabel("Gap before message (min, log scale)", fontsize=11)
    ax_gaps14.set_title("Gap sizes across 14-day arc\n(log scale — expect heavy tail)", fontsize=11)

    print("\n" + "=" * 60)
    print("14-DAY ARC SUMMARY")
    print("=" * 60)
    print(f"Total messages: {len(arc_messages)}")
    print(f"Messages/day breakdown:")
    for d in range(14):
        count = int(messages_per_day[d])
        phase = day_phases.get(d, "—")
        bar = "█" * min(count, 60)
        print(f"  Day {d+1:2d} [{phase:<15}]: {count:3d}  {bar}")

    plt.savefig("hawkes_validation.png", dpi=150, bbox_inches="tight")
    print("\nPlot saved to hawkes_validation.png")


if __name__ == "__main__":
    main()

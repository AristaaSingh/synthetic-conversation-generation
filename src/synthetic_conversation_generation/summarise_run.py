"""
Summarise generated conversation(s) — the headline metrics for a run.

Run with:
    python -m synthetic_conversation_generation.summarise_run <file.json>
    python -m synthetic_conversation_generation.summarise_run <directory>/

Accepts a single output file or a directory of them, so the same tool serves both
a one-off SLURM run and later corpus-level analysis.

The metrics chosen here are the ones that have actually distinguished a good run
from a bad one during development:

  span / median gap  — the arc (project_record.md 22, 23.8). A run that comes back
                       near 1.5 days with a ~1 minute median means phase-aware
                       session gaps are not firing.
  tension / incident — whether any VAWG dynamic accumulated at all (the long-
                       standing plateau-at-2 problem).
  intended vs realised categories — the gap between what the dialogue-flow planner
                       asked for (Beat.category) and what the assessor judged was
                       actually present (detected_categories). See 23.2: this
                       measures whether the generator delivered the plan.
  complete           — a mid-run checkpoint is otherwise indistinguishable from a
                       finished run (23.3). Never admit an incomplete run to the
                       corpus.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class RunSummary:
    path: Path
    complete: bool
    turns: int
    sessions: int
    span_days: float
    median_gap_min: float
    long_gaps: int              # gaps > 4h — session boundaries, which drive the arc
    tension: int
    phase: str
    incident: bool
    intended: set[str]          # categories the planner asked for
    realised: set[str]          # categories the assessor judged present
    persona_ok: bool = True

    @property
    def coverage(self) -> float:
        """Fraction of intended categories the assessor actually detected."""
        if not self.intended:
            return 0.0
        return len(self.intended & self.realised) / len(self.intended)

    @property
    def unplanned(self) -> set[str]:
        """Categories detected that were never planned."""
        return self.realised - self.intended


def summarise(path: Path) -> RunSummary:
    data = json.loads(path.read_text())

    ts = [datetime.strptime(m["timestamp"], "%Y-%m-%d %H:%M") for m in data["messages"]]
    if len(ts) > 1:
        gaps = sorted((ts[i + 1] - ts[i]).total_seconds() / 60 for i in range(len(ts) - 1))
        median = gaps[len(gaps) // 2]
        span_days = (ts[-1] - ts[0]).total_seconds() / 86400
        long_gaps = sum(1 for g in gaps if g > 240)
    else:
        median, span_days, long_gaps = 0.0, 0.0, 0

    intended = {
        b["category"]
        for flow in data.get("dialogue_flows", [])
        for b in flow.get("beats", [])
        if b.get("category")
    }
    realised = set(data["final_state"].get("detected_categories") or [])

    return RunSummary(
        path=path,
        complete=data.get("complete", True),
        turns=data.get("turns_generated", len(data["messages"])),
        sessions=len(data.get("dialogue_flows", [])),
        span_days=span_days,
        median_gap_min=median,
        long_gaps=long_gaps,
        tension=data["final_state"]["tension_level"],
        phase=data["final_state"]["phase"],
        incident=data["final_state"]["incident_occurred"],
        intended=intended,
        realised=realised,
    )


def print_one(s: RunSummary) -> None:
    d, h = int(s.span_days), int((s.span_days % 1) * 24)
    flag = "" if s.complete else "   <-- INCOMPLETE (checkpoint, not a finished run)"
    print(f"  complete:   {s.complete}{flag}")
    print(f"  turns:      {s.turns} | sessions: {s.sessions}")
    print(f"  span:       {d}d {h}h")
    print(f"  median gap: {s.median_gap_min:.1f} min | gaps >4h: {s.long_gaps}")
    print(f"  tension:    {s.tension}/5 | phase: {s.phase} | incident: {s.incident}")
    print(f"  categories: intended={sorted(s.intended) or '—'}")
    print(f"              realised={sorted(s.realised) or '—'}  "
          f"(coverage {s.coverage:.0%})")
    if s.unplanned:
        print(f"              unplanned={sorted(s.unplanned)}")


def print_corpus(summaries: list[RunSummary]) -> None:
    ok = [s for s in summaries if s.complete]
    print(f"{len(summaries)} runs ({len(ok)} complete, "
          f"{len(summaries) - len(ok)} incomplete)")
    print()
    print(f"  {'run':<28} {'turns':>5} {'sess':>4} {'span':>7} "
          f"{'med':>6} {'tens':>4} {'cov':>4}")
    print(f"  {'-'*28} {'-'*5} {'-'*4} {'-'*7} {'-'*6} {'-'*4} {'-'*4}")
    for s in sorted(summaries, key=lambda x: x.path.name):
        mark = " " if s.complete else "!"
        print(f" {mark}{s.path.stem[:28]:<28} {s.turns:>5} {s.sessions:>4} "
              f"{s.span_days:>6.1f}d {s.median_gap_min:>5.1f}m "
              f"{s.tension:>4} {s.coverage:>3.0%}")

    if ok:
        print()
        n = len(ok)
        print(f"  complete runs — mean span {sum(s.span_days for s in ok)/n:.1f}d | "
              f"mean tension {sum(s.tension for s in ok)/n:.1f}/5 | "
              f"incidents {sum(s.incident for s in ok)}/{n} | "
              f"mean coverage {sum(s.coverage for s in ok)/n:.0%}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Summarise generated conversation(s)")
    ap.add_argument("path", type=Path, help="Output JSON file, or a directory of them")
    args = ap.parse_args()

    if args.path.is_dir():
        files = sorted(args.path.glob("*.json"))
        if not files:
            print(f"No .json files in {args.path}")
            return 1
        print_corpus([summarise(f) for f in files])
    else:
        print_one(summarise(args.path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

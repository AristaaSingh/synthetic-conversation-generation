"""
Extract the SELFMA out-of-distribution evaluation set.

Run with:
    python -m synthetic_conversation_generation.evaluator.prepare_selfma \
        --data-dir "<path to SelfMA Dataset>" --output-dir data/evaluator

Why this set exists
-------------------
SELFMA is NEVER trained on. It is the independent test set, and it exists to solve
a specific problem: training the evaluator on Biasly and then evaluating with a
Biasly-trained classifier is circular — the classifier would be graded on the
distribution it learned. SELFMA is the only data in the project that is
simultaneously:

  * real       — human-reported, not synthetic and not movie-scripted
  * gendered   — 1,411 posts carry a gender tag
  * workplace  — a subset are set at work, matching the generation domain
  * dialogue   — speaker-labelled, multi-turn

Nothing else has all four. It therefore supports the strong claim ("a classifier
trained on movie subtitles correctly identifies real, human-reported workplace
microaggressions from a different corpus, collected a different way") rather than
the weak one ("it works on held-out data from its own distribution").

It is also the target distribution for the injector's evaluation (project_record.md
20.3): score real SELFMA dialogues, then pre- and post-fine-tune generated output,
and show the generated output moving toward the real.

Data notes (established by audit, project_record.md 19)
------------------------------------------------------
  * The annotation spreadsheet has NO gender tag; tags live only in the raw JSONL,
    so the two must be joined on Post ID.
  * Multi-turn dialogue exists ONLY in the `transcript` field of `type == "chat"`
    records — never in `quote`. Roughly 74% of transcript lines carry a
    "SPEAKER:: text" marker; the remainder are scene-setting narration, which is
    retained separately as context rather than mistaken for a turn.

References
----------
[1] Breitfeller, L., Ahn, E., Jurgens, D., & Tsvetkov, Y. (2019). Finding
    Microaggressions in the Wild: A Case for Locating Elusive Phenomena in Social
    Media Posts. EMNLP-IJCNLP 2019, pp. 1664-1674.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd

from synthetic_conversation_generation.data_models.microaggression_taxonomy import (
    CATEGORY_KEYS,
    is_valid,
)

# Tumblr moderator tags that mark a post as gender-related.
GENDER_TAGS = {"gender", "sexism", "misogyny", "sexual harassment", "women"}

# SELFMA sub-theme -> canonical (Capodilupo) category. See taxonomy_mapping.md §3.
#
# Note "Myth of Meritocracy": this is precisely the mechanism behind the incident in
# run 6664248 ("We're a meritocracy here; you just need to push harder"), and it is
# also the category Biasly can barely support (n=34 in the train split). SELFMA is
# therefore the only real-data check available for it.
SELFMA_TO_CANONICAL: dict[str, str] = {
    "Stereotype":            "traditional_gender_roles",
    "Myth of Meritocracy":   "assumptions_of_inferiority",
    "Second-Class Citizen":  "second_class_citizenship",
    "Ownership":             "second_class_citizenship",
    "Objectification":       "sexual_objectification",
    "Overt Aggression":      "use_of_sexist_language",
    "Denial of Lived Exp.":  "denial_of_reality_of_sexism",
}

# Sub-themes with no home in the Capodilupo scheme. Recorded rather than silently
# dropped: `Abnormality` and `Erasure` mapped to CHI's `pathologizing_character` and
# `exclusion`, both of which were removed when the taxonomy was revised (21.2)
# because Biasly could not support them. They are counted in the report so the loss
# is visible.
UNMAPPED_GENDER_SUBTHEMES = {"Abnormality", "Erasure"}

# Race-specific sub-themes — out of scope for a gender-only project.
RACE_SPECIFIC_SUBTHEMES = {"Criminal Status", "Alien in Own Land", "Monolith"}

ALL_SUBTHEMES = (list(SELFMA_TO_CANONICAL)
                 + sorted(UNMAPPED_GENDER_SUBTHEMES)
                 + sorted(RACE_SPECIFIC_SUBTHEMES))

# A dialogue is treated as workplace-set if any line mentions a work context. Kept
# deliberately simple and inspectable — the resulting subset is small enough to
# eyeball, and a false positive is more costly than a miss for an eval set.
WORKPLACE_RE = re.compile(
    r"\b(office|work|works|working|coworker|co-worker|colleague|boss|manager|"
    r"meeting|job|interview|client|engineer|company|team|desk|conference|"
    r"employee|staff|promotion|salary|intern)\b",
    re.IGNORECASE,
)

SPEAKER_RE = re.compile(r"^\s*([^:]{1,30}?)::\s*(.*)$", re.DOTALL)

# Fail loudly if the crosswalk drifts from the canonical taxonomy.
_unknown = {v for v in SELFMA_TO_CANONICAL.values() if not is_valid(v)}
if _unknown:
    raise ValueError(
        f"Crosswalk targets not in the canonical taxonomy: {sorted(_unknown)}. "
        f"Valid: {CATEGORY_KEYS}"
    )


def load_raw(path: Path) -> pd.DataFrame:
    """The raw scrape is JSONL (one object per line), not a JSON array."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    df["id"] = pd.to_numeric(df["id"], errors="coerce")
    return df


def is_gender(tags) -> bool:
    if not isinstance(tags, list):
        return False
    return any(str(t).strip().lower() in GENDER_TAGS for t in tags)


def parse_transcript(transcript) -> tuple[list[tuple[str, str]], list[str]]:
    """
    Split a transcript into (speaker, text) turns and free-standing narration.

    ~26% of lines have no "SPEAKER::" marker — they are scene-setting ("At the
    office. Three men are in a meeting...") or asides. Treating those as dialogue
    turns would corrupt the turn structure, so they are separated out and kept as
    context.
    """
    if not isinstance(transcript, list):
        return [], []
    turns, narration = [], []
    for line in transcript:
        s = str(line).strip()
        if not s:
            continue
        m = SPEAKER_RE.match(s)
        if m:
            speaker = m.group(1).strip()
            text = m.group(2).strip().strip('"')
            if text:
                turns.append((speaker, text))
        else:
            narration.append(s)
    return turns, narration


def format_dialogue(turns: list[tuple[str, str]]) -> str:
    return "\n".join(f"{s}: {t}" for s, t in turns)


def load_annotations(path: Path) -> pd.DataFrame:
    ann = pd.read_excel(path, header=1)
    ann = ann.loc[:, ~ann.columns.astype(str).str.startswith("Unnamed")]
    for c in ALL_SUBTHEMES:
        if c in ann.columns:
            ann[c] = pd.to_numeric(ann[c], errors="coerce").fillna(0)
    ann["Post ID"] = pd.to_numeric(ann["Post ID"], errors="coerce")
    return ann


def main() -> int:
    ap = argparse.ArgumentParser(description="Extract the SELFMA OOD evaluation set")
    ap.add_argument("--data-dir", required=True,
                    help="Directory holding microaggressions_v1.json and SelfMA Annotations.xlsx")
    ap.add_argument("--output-dir", default="data/evaluator")
    args = ap.parse_args()

    src = Path(args.data_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    raw = load_raw(src / "microaggressions_v1.json")
    ann = load_annotations(src / "SelfMA Annotations.xlsx")
    print(f"Loaded {len(raw):,} raw posts, {len(ann):,} annotation rows")

    raw["is_gender"] = raw["tags"].apply(is_gender)
    print(f"  gender-tagged: {raw.is_gender.sum():,}")

    # Multi-turn dialogue exists only on `chat` records.
    chat = raw[(raw["type"] == "chat") & raw["is_gender"]].copy()
    print(f"  gender + chat (real dialogues): {len(chat):,}")

    records = []
    for _, r in chat.iterrows():
        turns, narration = parse_transcript(r["transcript"])
        if len(turns) < 2:          # a single turn is not a dialogue
            continue
        full = " ".join([t for _, t in turns] + narration)
        records.append({
            "post_id": int(r["id"]),
            "dialogue": format_dialogue(turns),
            "narration": " ".join(narration),
            "n_turns": len(turns),
            "n_speakers": len({s for s, _ in turns}),
            "is_workplace": bool(WORKPLACE_RE.search(full)),
            "tags": "|".join(str(t) for t in r["tags"]) if isinstance(r["tags"], list) else "",
            "permalink": r.get("permalink", ""),
        })
    df = pd.DataFrame(records)
    print(f"  with >=2 parsed turns:          {len(df):,}")

    # Attach typology labels where the post was annotated (only the top ~1,300 were).
    ann_idx = ann.set_index("Post ID")
    canon_col, sub_col = [], []
    for pid in df["post_id"]:
        if pid not in ann_idx.index:
            canon_col.append("")
            sub_col.append("")
            continue
        row = ann_idx.loc[pid]
        if isinstance(row, pd.DataFrame):      # duplicate Post IDs
            row = row.iloc[0]
        subs = [c for c in ALL_SUBTHEMES if c in ann.columns and row.get(c, 0) == 1]
        canon = sorted({SELFMA_TO_CANONICAL[s] for s in subs if s in SELFMA_TO_CANONICAL})
        sub_col.append("|".join(subs))
        canon_col.append("|".join(canon))
    df["selfma_subthemes"] = sub_col
    df["canonical_categories"] = canon_col

    # This set is 100% positive — every SELFMA post IS a reported microaggression.
    # Recorded explicitly so it is never mistaken for a balanced set and used for
    # training: a classifier fitted on positives only learns to answer "yes".
    df["is_misogynistic"] = True

    df = df.sort_values(["is_workplace", "n_turns"], ascending=[False, False])
    df.to_csv(out / "selfma_eval.csv", index=False)

    wp = df[df.is_workplace]
    wp.to_csv(out / "selfma_eval_workplace.csv", index=False)

    print()
    print("=" * 66)
    print("SELFMA OOD EVALUATION SET")
    print("=" * 66)
    print(f"  all gender dialogues:  {len(df):,}   -> selfma_eval.csv")
    print(f"  workplace subset:      {len(wp):,}   -> selfma_eval_workplace.csv")
    print(f"  typology-labelled:     {(df.canonical_categories != '').sum():,}")
    print(f"  mean turns:            {df.n_turns.mean():.1f}  (range {df.n_turns.min()}-{df.n_turns.max()})")
    print()

    print("--- canonical category coverage (labelled dialogues) ---")
    c: Counter = Counter()
    for s in df["canonical_categories"]:
        if s:
            c.update(s.split("|"))
    for k in CATEGORY_KEYS:
        n = c.get(k, 0)
        print(f"  {n:4d}  {k}" + ("   <-- no real-data check available" if n == 0 else ""))
    print()

    print("--- sub-themes not representable in the canonical scheme (21.2) ---")
    u: Counter = Counter()
    for s in df["selfma_subthemes"]:
        for t in (s.split("|") if s else []):
            if t in UNMAPPED_GENDER_SUBTHEMES:
                u[t] += 1
    for k, n in u.most_common():
        print(f"  {n:4d}  {k}  (was CHI '{'pathologizing_character' if k == 'Abnormality' else 'exclusion'}')")
    if not u:
        print("  (none present in this subset)")
    print()

    print("--- sample workplace dialogue ---")
    if len(wp):
        s = wp.iloc[0]
        print(f"  post {s.post_id} | {s.n_turns} turns | {s.canonical_categories or 'unlabelled'}")
        for line in s.dialogue.split("\n")[:4]:
            print(f"    {line[:100]}")

    print(f"\nWrote to {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

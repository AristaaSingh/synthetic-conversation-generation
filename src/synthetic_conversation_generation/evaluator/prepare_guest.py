"""
Preprocess the Guest et al. expert-annotated online misogyny dataset.

Guest's primary value is its HARD NEGATIVES. Its taxonomy includes a
`Nonmisogynistic_personal_attack` class — text that is hostile and derogatory but
NOT misogynistic. This is exactly the boundary the evaluator most needs: the
perpetrator in the generation pipeline (Ryan) is terse and dismissive without
being overtly sexist, and a classifier that cannot separate "rude" from
"misogynistic" would flag his every curt reply, making the evaluation noise.

Use `final_labels.csv` (adjudicated, one label per post), NOT `original_labels.csv`
(raw per-annotator judgements — useful only for studying annotator disagreement).

On the category crosswalk: Guest's level_2 classifies misogyny by LINGUISTIC FORM
(Derogation, Pejorative, Personal attack, Treatment), which is orthogonal to the
Capodilupo scheme's SOCIAL MECHANISMS (inferiority, roles, objectification). Only
the two clearly-lexical classes are mapped; the rest contribute to the binary head
only. Forcing the ambiguous ones would inject label noise.

Run with:
    python -m synthetic_conversation_generation.evaluator.prepare_guest \
        --input "<path>/final_labels.csv" --output-dir data/evaluator

References
----------
[1] Guest, E., Vidgen, B., Mittos, A., Sastry, N., Tyson, G., & Margetts, H.
    (2021). An Expert Annotated Dataset for the Detection of Online Misogyny.
    EACL 2021, pp. 1336-1350.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from synthetic_conversation_generation.data_models.microaggression_taxonomy import (
    CATEGORY_KEYS,
    is_valid,
)
from synthetic_conversation_generation.evaluator.dataset_common import deduplicate

# Only the two unambiguously-lexical Guest classes are mapped. Both are about
# hostile/derogatory language, which is `use_of_sexist_language` in the canonical
# scheme. Treatment / Personal_attack are left unmapped (binary only) because they
# describe form, not the Capodilupo mechanism.
GUEST_L2_TO_CANONICAL: dict[str, str] = {
    "Derogation":             "use_of_sexist_language",
    "Misogynistic_pejorative": "use_of_sexist_language",
}

# level_2 values that are NOT misogynistic. `Nonmisogynistic_personal_attack` is the
# prize: rude-but-not-sexist, the hard-negative class.
HARD_NEGATIVE_L2 = "Nonmisogynistic_personal_attack"

_unknown = {v for v in GUEST_L2_TO_CANONICAL.values() if not is_valid(v)}
if _unknown:
    raise ValueError(f"Crosswalk targets not canonical: {sorted(_unknown)}. Valid: {CATEGORY_KEYS}")


def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["text"] = df["body"].astype(str).str.strip()
    df["is_misogynistic"] = df["level_1"].astype(str).str.strip() == "Misogynistic"
    df["is_hard_negative"] = df["level_2"].astype(str).str.strip() == HARD_NEGATIVE_L2
    df = df[df["text"].str.len() > 0]
    return df


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse to one row per post (entry_id).

    final_labels has ~184 more rows than unique posts because a post can carry
    multiple level_2 tags. A post is misogynistic if ANY of its rows says so; its
    canonical categories are the union of the mapped ones.

    "any misogynistic wins" is the conservative rule and matches Biasly's
    ">=1 annotator" philosophy (preserve the judgement that something IS
    misogynistic). Exactly ONE post in the dataset carries both a Misogynistic and
    a Nonmisogynistic (Counter_speech) tag — a counter-speech post that quotes a
    pejorative — and this rule labels it misogynistic. n=1, negligible, documented.
    """
    rows = []
    for eid, grp in df.groupby("entry_id"):
        is_mis = bool(grp["is_misogynistic"].any())
        is_hard_neg = bool(grp["is_hard_negative"].any()) and not is_mis
        canon = sorted({
            GUEST_L2_TO_CANONICAL[l2]
            for l2 in grp["level_2"].astype(str).str.strip()
            if l2 in GUEST_L2_TO_CANONICAL
        })
        rows.append({
            "source": "guest",
            "text": grp["text"].iloc[0],
            "is_misogynistic": is_mis,
            "severity": pd.NA,
            "canonical_categories": "|".join(canon),
            "in_scope": is_mis,
            "is_hard_negative": is_hard_neg,
            "rationale": "",
        })
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description="Preprocess Guest et al. for the evaluator")
    ap.add_argument("--input", required=True, help="Path to final_labels.csv")
    ap.add_argument("--output-dir", default="data/evaluator")
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = load(Path(args.input))
    agg = aggregate(df)
    # Reddit reposts / copypasta produce identical body text across distinct posts.
    agg = deduplicate(agg, source="guest")
    agg.to_csv(out / "guest_all.csv", index=False)

    print(f"Guest: {len(agg):,} posts")
    print(f"  misogynistic:      {int(agg.is_misogynistic.sum()):,}")
    print(f"  non-misogynistic:  {int((~agg.is_misogynistic).sum()):,}")
    print(f"  HARD NEGATIVES (rude, not sexist): {int(agg.is_hard_negative.sum()):,}")
    print(f"  with a mapped category:            {int((agg.canonical_categories != '').sum()):,}")
    print(f"Wrote {out / 'guest_all.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

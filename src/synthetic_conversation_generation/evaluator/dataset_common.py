"""
Shared contract and utilities for evaluator dataset preparation.

Every prepare_*.py emits a CSV with the columns in COMMON_COLUMNS, so the trainer
can concatenate them without per-source special-casing. Partial labels are
expected and encoded as blanks / NA (see below), NOT as zeros.

The `in_scope` column is a Biasly-specific, Objective-B (generation) concept: it
marks datapoints whose microaggression category is in scope for the *generator*
(the violent Biasly classes are excluded there). It is meaningless for the
evaluator and for the binary-only datasets, where it simply mirrors
is_misogynistic. THE EVALUATOR TRAINER MUST NOT FILTER ON in_scope — the evaluator
wants all data, including the violent classes that anchor the top of the severity
scale.
"""
from __future__ import annotations

import re

import pandas as pd

COMMON_COLUMNS = [
    "source",                 # dataset name
    "text",                   # the statement / utterance
    "is_misogynistic",        # bool — the binary label (present for every source)
    "severity",               # 0-1000, Biasly only; NA elsewhere (skip severity loss)
    "canonical_categories",   # "|"-joined canonical keys; "" where unlabelled
    "in_scope",               # Biasly/Objective-B flag; see module docstring
    "rationale",              # expert rationale, Biasly only; "" elsewhere
]


def _normalise(text: str) -> str:
    """Lower-case, collapse whitespace — for exact-duplicate detection only."""
    return re.sub(r"\s+", " ", str(text).lower().strip())


def deduplicate(df: pd.DataFrame, text_col: str = "text",
                label_col: str = "is_misogynistic", source: str = "") -> pd.DataFrame:
    """
    Remove exact-duplicate texts, and drop any text whose duplicates disagree on
    the label.

    Rationale:
      * Same text with the SAME label -> keep one copy. Duplicates inflate the
        example's effective weight and, if a train/val split is ever taken within
        a source, would leak across it.
      * Same text with DIFFERENT labels -> drop every copy. The label is
        unreliable, and there is no principled way to pick a side. (Only CMSB has
        these, from its overlapping sub-datasets.)

    Returns the deduplicated frame and prints what was removed, so the loss is
    never silent.
    """
    n0 = len(df)
    key = df[text_col].map(_normalise)

    # Identify texts with conflicting labels across their duplicates.
    label_variety = df.groupby(key)[label_col].transform("nunique")
    conflicting = label_variety > 1
    n_conflict_rows = int(conflicting.sum())
    n_conflict_groups = int(df.loc[conflicting].groupby(key.loc[conflicting]).ngroups) if n_conflict_rows else 0

    kept = df.loc[~conflicting].copy()
    kept_key = kept[text_col].map(_normalise)
    before_dedup = len(kept)
    kept = kept.loc[~kept_key.duplicated()].copy()
    n_redundant = before_dedup - len(kept)

    tag = f"[{source}] " if source else ""
    print(f"  {tag}dedup: {n0} -> {len(kept)} "
          f"(dropped {n_redundant} same-label duplicates, "
          f"{n_conflict_rows} rows in {n_conflict_groups} conflicting-label groups)")
    return kept

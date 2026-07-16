"""
Preprocess the Biasly dataset into clean train/val/test splits for the evaluator.

Applies the data-hygiene rules established in `taxonomy_mapping.md` and aggregates
the 3-annotator rows down to one row per datapoint.

References
----------
[1] Sheppard, B., Richter, A., Cohen, A., Smith, E. A., Kneese, T., Pelletier, C.,
    Baldini, I., & Dong, Y. (2024). Biasly: An Expert-Annotated Dataset for Subtle
    Misogyny Detection and Mitigation. Findings of ACL 2024, pp. 427-452.

Usage
-----
    python -m synthetic_conversation_generation.evaluator.prepare_biasly \
        --input  "<path to biasly_dataset.csv>" \
        --output-dir data/evaluator
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import pandas as pd

from synthetic_conversation_generation.data_models.microaggression_taxonomy import (
    CATEGORY_KEYS,
    is_valid,
)

# --------------------------------------------------------------------------
# Hygiene rule 2: annotation-UI artefacts that are not categories at all.
# --------------------------------------------------------------------------
ARTEFACT_LABELS = {
    "add optional explanation",
    "",
    "nan",
}

# --------------------------------------------------------------------------
# Biasly's 12 categories. The raw strings carry parenthetical glosses and vary
# slightly between annotation rounds, so we match on a distinctive substring
# rather than the full string.
#
# Hygiene rule 3: the two sexualization variants
#   "Sexualization (focus on appearance, degrading language)"  and
#   "Objectification/sexualization (focus on appearance)"
# are a mid-annotation taxonomy revision, not distinct classes -> both normalise
# to `sexualization`.
# --------------------------------------------------------------------------
_BIASLY_PATTERNS: list[tuple[str, str]] = [
    ("trivialization",   "trivialization"),
    ("gender essentialism", "gender_essentialism"),
    ("objectification/sexualization", "sexualization"),   # must precede "sexualization"
    ("sexualization",    "sexualization"),
    ("lacking autonomy", "lacking_autonomy"),
    ("gendered slur",    "gendered_slurs"),
    ("dehumanization",   "dehumanization"),
    ("domestic violence", "domestic_violence"),
    ("rape",             "sexual_violence"),
    ("phallocentrism",   "phallocentrism"),
    ("intersectional",   "intersectional"),
    ("anti-feminism",    "anti_feminism"),
    ("transmisogyny",    "transmisogyny_homophobia"),
]

# --------------------------------------------------------------------------
# Crosswalk: Biasly's 12 inductive categories -> the canonical taxonomy.
#
# The canonical categories themselves live in
# `data_models/microaggression_taxonomy.py` -- the single source of truth. Only
# the Biasly-specific *mapping* belongs here.
#
# NOTE: categories marked OUT_OF_SCOPE are excluded from Objective B (generation)
# because they would push the generator off-domain, but are RETAINED for
# Objective C (the evaluator) because they anchor the top of the 0-1000 severity
# scale. Removing them would compress the severity distribution and degrade the
# regressor. Hence `in_scope` is a flag, not a filter.
# --------------------------------------------------------------------------
BIASLY_TO_CANONICAL: dict[str, str] = {
    # Biasly: "Infantilizing or paternalistic language, women are not taken seriously"
    "trivialization":      "assumptions_of_inferiority",
    # Biasly: "women are good at childrearing... women are untrustworthy and overly emotional"
    # (Ambiguous: also carries pathologizing content. Primary mapping recorded.)
    "gender_essentialism": "traditional_gender_roles",
    # Biasly: "Women are not able to make decisions or must defer to male authorities"
    "lacking_autonomy":    "second_class_citizenship",
    # Biasly: "Outsized focus on appearance, degrading language"
    "sexualization":       "sexual_objectification",
    # Biasly: "Chick, b*tch, c*nt, etc."
    "gendered_slurs":      "use_of_sexist_language",
    # Biasly: "Comparing women to animals or objects"
    "dehumanization":      "use_of_sexist_language",
    # Weak mapping: anti-feminism ("women shouldn't have equal rights") is not the
    # same as denying that sexism is real, but it is the nearest Capodilupo theme.
    # Flagged as low-confidence; n=52, so per-class metrics will be unreliable.
    "anti_feminism":       "denial_of_reality_of_sexism",
}

OUT_OF_SCOPE = {
    "domestic_violence",
    "sexual_violence",
    "phallocentrism",
    "intersectional",
    "transmisogyny_homophobia",
    "other",
}

# Fail loudly if this crosswalk ever drifts from the canonical taxonomy -- e.g. if
# a category is renamed in microaggression_taxonomy.py but not here. Silent
# mismatch would produce empty label columns rather than an error.
_unknown = {v for v in BIASLY_TO_CANONICAL.values() if not is_valid(v)}
if _unknown:
    raise ValueError(
        f"Crosswalk targets are not in the canonical taxonomy: {sorted(_unknown)}. "
        f"Valid keys: {CATEGORY_KEYS}"
    )


def normalise_category(raw: str) -> str | None:
    """Map one raw Biasly label onto its normalised key, or None if an artefact."""
    s = str(raw).strip().strip("[]'\"").lower()
    if s in ARTEFACT_LABELS:
        return None
    for pattern, key in _BIASLY_PATTERNS:
        if pattern in s:
            return key
    if s.startswith("other") or "resemtment" in s or "possessive" in s:
        return "other"
    return "other"


def parse_categories(cell) -> list[str]:
    """
    Hygiene rule 1: split on ';' ONLY.

    Two Biasly categories contain commas inside their own names, e.g.
    "Sexualization (focus on appearance, degrading language)". Splitting on
    commas silently shatters them into fragments.
    """
    if pd.isna(cell):
        return []
    out = []
    for part in str(cell).split(";"):
        key = normalise_category(part)
        if key is not None:
            out.append(key)
    return sorted(set(out))


def aggregate_to_datapoints(df: pd.DataFrame) -> pd.DataFrame:
    """
    Hygiene rule 4: collapse the 3 annotator rows per datapoint into one.

    Following the Biasly paper: a datapoint is misogynistic if AT LEAST ONE
    annotator labelled it so. This deliberately preserves minority annotator
    judgements rather than majority-voting them away -- important because the
    dataset targets *subtle* misogyny, where disagreement is signal, not noise.
    """
    rows = []
    for dp_id, grp in df.groupby("datapoint_id"):
        is_mis = (grp["is_misogynistic"] == "Yes").any()

        # Severity: mean across only those annotators who judged it misogynistic.
        sev_vals = pd.to_numeric(
            grp.loc[grp["is_misogynistic"] == "Yes", "original_severity"],
            errors="coerce",
        ).dropna()
        severity = float(sev_vals.mean()) if len(sev_vals) else 0.0

        # Categories: union across annotators (mirrors the >=1 rule above).
        cats: list[str] = []
        for cell in grp["misogynistic_inferences"]:
            cats.extend(parse_categories(cell))
        cats = sorted(set(cats))

        canonical = sorted({BIASLY_TO_CANONICAL[c] for c in cats if c in BIASLY_TO_CANONICAL})
        in_scope = bool(canonical) and not all(c in OUT_OF_SCOPE for c in cats)

        # Expert rationales -- the annotator naming the implicit belief conveyed.
        # Valuable as a conditioning signal for the injector (see project_record.md 20.3).
        rationales = [str(x) for x in grp["inferences_explanation"].dropna()]

        rows.append({
            "datapoint_id": dp_id,
            "text": grp["datapoint"].iloc[0],
            "is_misogynistic": bool(is_mis),
            "severity": severity,
            "biasly_categories": "|".join(cats),
            "canonical_categories": "|".join(canonical),
            "in_scope": in_scope,
            "n_annotators_yes": int((grp["is_misogynistic"] == "Yes").sum()),
            "rationale": rationales[0] if rationales else "",
        })
    return pd.DataFrame(rows)


def stratified_split(df: pd.DataFrame, seed: int = 42) -> dict[str, pd.DataFrame]:
    """
    80/10/10 split, stratified on the binary label, matching the Biasly paper.

    Splitting is done at DATAPOINT level (post-aggregation), so the same text can
    never appear in two splits via its sibling annotator rows.
    """
    rng = pd.Series(range(len(df))).sample(frac=1.0, random_state=seed).values
    df = df.iloc[rng].reset_index(drop=True)

    parts: dict[str, list[pd.DataFrame]] = {"train": [], "val": [], "test": []}
    for _, grp in df.groupby("is_misogynistic"):
        n = len(grp)
        n_train, n_val = int(0.8 * n), int(0.1 * n)
        parts["train"].append(grp.iloc[:n_train])
        parts["val"].append(grp.iloc[n_train:n_train + n_val])
        parts["test"].append(grp.iloc[n_train + n_val:])

    return {
        k: pd.concat(v).sample(frac=1.0, random_state=seed).reset_index(drop=True)
        for k, v in parts.items()
    }


def build_rewrite_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract the parallel rewrite corpus (misogynistic <-> neutral).

    Reversed, these are the injector's training signal: (neutral + target) -> misogynistic.
    Kept at ANNOTATION level, not datapoint level: multiple annotators rewrote the
    same text differently, and each rewrite is a legitimate demonstration. The
    Biasly paper treats each rewrite as its own datapoint too -- but note this means
    2,977 rewrites cover only ~1,985 unique datapoints.
    """
    rw = df[df["rewrite"].notna()].copy()
    rw["categories"] = rw["misogynistic_inferences"].apply(lambda c: "|".join(parse_categories(c)))
    rw["canonical"] = rw["categories"].apply(
        lambda s: "|".join(sorted({BIASLY_TO_CANONICAL[c]
                                   for c in s.split("|") if c in BIASLY_TO_CANONICAL}))
    )
    return pd.DataFrame({
        "datapoint_id": rw["datapoint_id"],
        "misogynistic_text": rw["datapoint"],
        "neutral_text": rw["rewrite"],
        "severity": pd.to_numeric(rw["original_severity"], errors="coerce"),
        "rewrite_severity": pd.to_numeric(rw["rewrite_severity"], errors="coerce"),
        "biasly_categories": rw["categories"],
        "canonical_categories": rw["canonical"],
        "rationale": rw["inferences_explanation"].fillna(""),
    })


def main():
    ap = argparse.ArgumentParser(description="Preprocess Biasly for the evaluator")
    ap.add_argument("--input", required=True, help="Path to biasly_dataset.csv")
    ap.add_argument("--output-dir", default="data/evaluator")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(args.input, index_col=0)
    print(f"Loaded {len(raw):,} annotations over {raw.datapoint_id.nunique():,} datapoints")

    # --- true category distribution (hygiene rule 1 applied) ---
    counter: Counter = Counter()
    for cell in raw["misogynistic_inferences"]:
        counter.update(parse_categories(cell))
    print("\n--- Category distribution (split on ';' only, artefacts dropped) ---")
    for cat, n in counter.most_common():
        tag = "" if cat in BIASLY_TO_CANONICAL else "   [out of scope]"
        print(f"  {n:6,d}  {cat}{tag}")

    # --- aggregate ---
    dp = aggregate_to_datapoints(raw)
    print(f"\n--- Aggregated to {len(dp):,} datapoints ---")
    print(f"  misogynistic (>=1 annotator): {dp.is_misogynistic.sum():,} "
          f"({100 * dp.is_misogynistic.mean():.1f}%)")
    print(f"  clean:                        {(~dp.is_misogynistic).sum():,}")
    print(f"  in-scope (for Objective B):   {dp.in_scope.sum():,}")
    print(f"  with an expert rationale:     {(dp.rationale != '').sum():,}")

    sev = dp.loc[dp.is_misogynistic, "severity"]
    print(f"  severity  mean={sev.mean():.1f}  sd={sev.std():.1f}  "
          f"min={sev.min():.0f}  max={sev.max():.0f}")

    print("\n--- Canonical category distribution (misogynistic datapoints) ---")
    ccount: Counter = Counter()
    for s in dp.loc[dp.is_misogynistic, "canonical_categories"]:
        if s:
            ccount.update(s.split("|"))
    for cat, n in ccount.most_common():
        print(f"  {n:6,d}  {cat}")

    # --- splits ---
    splits = stratified_split(dp, seed=args.seed)
    print("\n--- Splits ---")
    for name, part in splits.items():
        part.to_csv(out / f"biasly_{name}.csv", index=False)
        print(f"  {name:5s} {len(part):5,d}  "
              f"({part.is_misogynistic.sum():,} misogynistic, "
              f"{100 * part.is_misogynistic.mean():.1f}%)")

    # --- rewrite pairs (for the injector) ---
    pairs = build_rewrite_pairs(raw)
    pairs.to_csv(out / "biasly_rewrite_pairs.csv", index=False)
    print(f"\n--- Rewrite pairs ---")
    print(f"  {len(pairs):,} pairs over {pairs.datapoint_id.nunique():,} unique datapoints")
    print(f"  with a rationale: {(pairs.rationale != '').sum():,}")

    print(f"\nWrote to {out.resolve()}")


if __name__ == "__main__":
    main()

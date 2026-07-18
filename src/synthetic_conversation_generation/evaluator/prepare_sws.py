"""
Preprocess the Sexist Workplace Statements (SWS) dataset.

SWS is the ONLY workplace-register dataset in the pool. Its role is to teach the
binary head what workplace sexism looks like in the register the generator
produces — Biasly is movie subtitles, CMSB is tweets, Guest is Reddit. It is
binary only: no category labels, no severity, so it contributes to the binary
head alone.

Run with:
    python -m synthetic_conversation_generation.evaluator.prepare_sws \
        --input "<path>/ISEP Sexist Data labeling.xlsx" --output-dir data/evaluator

References
----------
[1] Grosz, D., & Conde-Cespedes, P. (2020). Automatic Detection of Sexist
    Statements Commonly Used at the Workplace. PAKDD 2020 LDRC Workshop.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from synthetic_conversation_generation.evaluator.dataset_common import deduplicate


def load(path: Path) -> pd.DataFrame:
    """SWS ships as a one-sheet .xlsx (a .tsv exported into Excel)."""
    df = pd.read_excel(path)
    df = df.rename(columns={"Sentences": "text", "Label": "label"})
    df["text"] = df["text"].astype(str).str.strip()
    df["is_misogynistic"] = pd.to_numeric(df["label"], errors="coerce") == 1
    df = df[df["text"].str.len() > 0].dropna(subset=["is_misogynistic"])
    return df


def to_common_format(df: pd.DataFrame) -> pd.DataFrame:
    """Emit the column set shared by every evaluator training file.

    severity, canonical_categories and rationale are empty: SWS carries none of
    them. The trainer must therefore skip the severity/category losses for these
    rows (partial labels), not treat the blanks as zeros.
    """
    return pd.DataFrame({
        "source": "sws",
        "text": df["text"].values,
        "is_misogynistic": df["is_misogynistic"].values,
        "severity": pd.NA,
        "canonical_categories": "",
        "in_scope": df["is_misogynistic"].values,
        "rationale": "",
    })


def main() -> int:
    ap = argparse.ArgumentParser(description="Preprocess SWS for the evaluator")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output-dir", default="data/evaluator")
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = load(Path(args.input))
    common = to_common_format(df)
    common = deduplicate(common, source="sws")
    common.to_csv(out / "sws_all.csv", index=False)

    print(f"SWS: {len(common):,} statements")
    print(f"  sexist:     {int(common.is_misogynistic.sum()):,}")
    print(f"  non-sexist: {int((~common.is_misogynistic).sum()):,}")
    print(f"  (binary only — no severity or category labels)")
    print(f"Wrote {out / 'sws_all.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

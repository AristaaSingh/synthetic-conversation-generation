"""
Preprocess the "Call me sexist, but..." (CMSB) dataset.

CMSB contributes two things:

  1. Binary training data for the evaluator: 1,809 sexist / 11,822 not — a large
     pool of negatives, grounded in a review of 30 psychological sexism scales
     (the same theoretical lineage as WoMenS). Binary only: no category, no
     severity, so it feeds the binary head alone.

  2. A SECOND parallel corpus for the injector (Objective B), independent of
     Biasly. CMSB built adversarial examples by making MINIMAL edits to sexist
     statements to render them non-sexist. Each such row links to its source via
     `of_id`. Reversed (non-sexist -> sexist), these are demonstrations of
     *producing* sexism at the level of a single surgical edit — exactly the
     injector's training signal, and in a different register (tweets) from
     Biasly's movie subtitles.

Run with:
    python -m synthetic_conversation_generation.evaluator.prepare_cmsb \
        --input "<path>/sexism_data.csv" --output-dir data/evaluator

References
----------
[1] Samory, M., Sen, I., Kohne, J., Floeck, F., & Wagner, C. (2021). "Call me
    sexist, but...": Revisiting Sexism Detection Using Psychological Scales and
    Adversarial Samples. ICWSM 2021, pp. 573-584.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from synthetic_conversation_generation.evaluator.dataset_common import deduplicate


def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["text"] = df["text"].astype(str).str.strip()
    # `sexist` is a Python-bool-like string/bool depending on the reader.
    df["is_misogynistic"] = df["sexist"].astype(str).str.lower().isin(["true", "1"])
    df = df[df["text"].str.len() > 0]
    return df


def to_common_format(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "source": "cmsb",
        "text": df["text"].values,
        "is_misogynistic": df["is_misogynistic"].values,
        "severity": pd.NA,
        "canonical_categories": "",
        "in_scope": df["is_misogynistic"].values,
        "rationale": "",
    })


def build_adversarial_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reconstruct (neutral -> sexist) pairs from CMSB's adversarial examples.

    An adversarial row (of_id != -1) is a minimally-edited NEUTRAL version of the
    sexist source row it points to via `of_id`. To get an injection pair we join
    the adversarial (neutral) text to its source (sexist) text.
    """
    by_id = df.set_index("id")
    pairs = []
    adversarial = df[df["of_id"] != -1]
    for _, row in adversarial.iterrows():
        src_id = row["of_id"]
        if src_id not in by_id.index:
            continue
        src = by_id.loc[src_id]
        if isinstance(src, pd.DataFrame):
            src = src.iloc[0]
        # Only keep the pair if the source really is the sexist side and the
        # adversarial row is the neutral side — the edit direction we want.
        if bool(src["is_misogynistic"]) and not bool(row["is_misogynistic"]):
            pairs.append({
                "source": "cmsb",
                "misogynistic_text": src["text"],
                "neutral_text": row["text"],
                "severity": pd.NA,          # CMSB has no severity scale
                "canonical_categories": "",
            })
    return pd.DataFrame(pairs)


def main() -> int:
    ap = argparse.ArgumentParser(description="Preprocess CMSB for evaluator + injector")
    ap.add_argument("--input", required=True, help="Path to sexism_data.csv")
    ap.add_argument("--output-dir", default="data/evaluator")
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = load(Path(args.input))

    common = to_common_format(df)
    common = deduplicate(common, source="cmsb")
    common.to_csv(out / "cmsb_all.csv", index=False)
    print(f"CMSB (evaluator): {len(common):,} texts")
    print(f"  sexist:     {int(common.is_misogynistic.sum()):,}")
    print(f"  non-sexist: {int((~common.is_misogynistic).sum()):,}")

    pairs = build_adversarial_pairs(df)
    pairs.to_csv(out / "cmsb_pairs.csv", index=False)
    print(f"CMSB (injector): {len(pairs):,} adversarial (neutral -> sexist) pairs")
    print(f"Wrote to {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

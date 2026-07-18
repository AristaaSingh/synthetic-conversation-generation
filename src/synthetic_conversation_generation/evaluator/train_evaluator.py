"""
Train the misogyny evaluator (Objective C).

    python -m synthetic_conversation_generation.evaluator.train_evaluator \
        --data-dir data/evaluator --output-dir models/evaluator --epochs 3

A shared DeBERTa encoder with binary + severity + category heads (see model.py),
trained on the pooled Biasly/CMSB/Guest/SWS data and evaluated on TWO held-out
sets:
  * biasly_test.csv  — in-distribution (same family as most training data)
  * selfma_eval.csv  — OUT-of-distribution: real, human-reported workplace
                       microaggression dialogue, never trained on. This is the
                       claim that matters — does a model trained on movie
                       subtitles / tweets recognise real workplace microaggressions?

Key correctness points, each from evaluator.md:
  * pool-time dedup removes the 12 CMSB/SWS shared tweets;
  * training does NOT filter on `in_scope` (a generation-side flag);
  * severity/category losses are masked to rows that carry those labels;
  * class weighting (not resampling) handles the 19% positive rate;
  * SELFMA is positive-only, so only recall / catch-rate is meaningful there.

A --smoke flag runs the whole path on a tiny model and a data subset (no GPU) to
verify correctness in seconds.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from synthetic_conversation_generation.data_models.microaggression_taxonomy import CATEGORY_KEYS
from synthetic_conversation_generation.evaluator.dataset_common import deduplicate
from synthetic_conversation_generation.evaluator.model import MultiHeadEvaluator, N_CATEGORIES

# Training sources pooled for the evaluator. Biasly val/test are held out; SELFMA
# is never here.
TRAIN_FILES = ["biasly_train.csv", "cmsb_all.csv", "guest_all.csv", "sws_all.csv"]
CAT_INDEX = {k: i for i, k in enumerate(CATEGORY_KEYS)}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def _encode_categories(cell) -> tuple[np.ndarray, bool]:
    """(multi-hot vector, has_label). Blank -> zero vector, has_label=False."""
    vec = np.zeros(N_CATEGORIES, dtype=np.float32)
    if not isinstance(cell, str) or not cell.strip():
        return vec, False
    for k in cell.split("|"):
        if k in CAT_INDEX:
            vec[CAT_INDEX[k]] = 1.0
    return vec, True


def load_pool(data_dir: Path) -> pd.DataFrame:
    frames = []
    for f in TRAIN_FILES:
        p = data_dir / f
        if not p.exists():
            print(f"  WARNING: {f} missing — skipping")
            continue
        frames.append(pd.read_csv(p))
    pool = pd.concat(frames, ignore_index=True)
    pool["text"] = pool["text"].astype(str)
    # Pool-time dedup: removes the 12 identical CMSB/SWS tweets (and any other
    # cross-source exact matches). Conflicting-label matches are dropped.
    pool = deduplicate(pool, source="pool")
    return pool.reset_index(drop=True)


class EvaluatorDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_len: int):
        self.texts = df["text"].tolist()
        self.binary = df["is_misogynistic"].astype(float).to_numpy()
        # severity: NaN where absent (drives the masked severity loss).
        self.severity = pd.to_numeric(df.get("severity"), errors="coerce").to_numpy(dtype=np.float32) \
            if "severity" in df else np.full(len(df), np.nan, np.float32)
        cats, masks = zip(*(_encode_categories(c) for c in df.get("canonical_categories", [""] * len(df))))
        self.category = np.stack(cats)
        self.category_mask = np.array(masks)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        enc = self.tokenizer(
            self.texts[i], truncation=True, max_length=self.max_len,
            padding="max_length", return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "binary_label": torch.tensor(self.binary[i], dtype=torch.float),
            "severity_label": torch.tensor(self.severity[i], dtype=torch.float),
            "category_label": torch.tensor(self.category[i], dtype=torch.float),
            "category_mask": torch.tensor(bool(self.category_mask[i])),
        }


def load_selfma(data_dir: Path, tokenizer, max_len: int) -> EvaluatorDataset | None:
    """SELFMA dialogues as an eval set. Text is the joined dialogue; all positive."""
    p = data_dir / "selfma_eval.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    df = pd.DataFrame({
        "text": df["dialogue"].astype(str),
        "is_misogynistic": True,
        "canonical_categories": df.get("canonical_categories", ""),
    })
    return EvaluatorDataset(df, tokenizer, max_len)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def binary_metrics(y_true, logits) -> dict:
    from sklearn.metrics import f1_score, precision_score, recall_score
    pred = (torch.sigmoid(torch.tensor(logits)) > 0.5).int().numpy()
    return {
        "precision": round(precision_score(y_true, pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, pred, zero_division=0), 4),
        "macro_f1": round(f1_score(y_true, pred, average="macro", zero_division=0), 4),
    }


def category_metrics(y_true, logits) -> dict:
    """Per-class F1 — reported per class because support is highly uneven."""
    from sklearn.metrics import f1_score
    pred = (torch.sigmoid(torch.tensor(logits)) > 0.5).int().numpy()
    y_true = np.asarray(y_true)
    out = {}
    for i, k in enumerate(CATEGORY_KEYS):
        support = int(y_true[:, i].sum())
        out[k] = {"f1": round(f1_score(y_true[:, i], pred[:, i], zero_division=0), 4),
                  "support": support}
    return out


def severity_metrics(y_true, y_pred) -> dict:
    mask = ~np.isnan(y_true)
    if mask.sum() < 2:
        return {}
    yt, yp = y_true[mask], y_pred[mask]
    mae = float(np.mean(np.abs(yt - yp)))
    corr = float(np.corrcoef(yt, yp)[0, 1]) if np.std(yp) > 0 else 0.0
    return {"mae": round(mae, 1), "pearson_r": round(corr, 4)}


@torch.no_grad()
def evaluate(model, loader, device, with_category=True) -> dict:
    model.eval()
    bl, bt, sl, st, cl, ct, cm = [], [], [], [], [], [], []
    for batch in loader:
        out = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
        )
        bl.append(out.binary_logit.cpu().numpy()); bt.append(batch["binary_label"].numpy())
        st.append(out.severity.cpu().numpy()); sl.append(batch["severity_label"].numpy())
        ct.append(out.category_logits.cpu().numpy()); cl.append(batch["category_label"].numpy())
        cm.append(batch["category_mask"].numpy())
    bl, bt = np.concatenate(bl), np.concatenate(bt)
    result = {"binary": binary_metrics(bt, bl)}
    result["severity"] = severity_metrics(np.concatenate(sl), np.concatenate(st))
    if with_category:
        cl, ct, cm = np.concatenate(cl), np.concatenate(ct), np.concatenate(cm)
        if cm.any():
            result["category"] = category_metrics(cl[cm], ct[cm])
    return result


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def pick_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> int:
    ap = argparse.ArgumentParser(description="Train the misogyny evaluator")
    ap.add_argument("--data-dir", type=Path, default=Path("data/evaluator"))
    ap.add_argument("--output-dir", type=Path, default=Path("models/evaluator"))
    ap.add_argument("--model-name", default="microsoft/deberta-v3-base",
                    help="Encoder. roberta-base is a no-sentencepiece fallback.")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--smoke", action="store_true",
                    help="Tiny model + 200 rows + 1 epoch, to verify the path (no GPU).")
    args = ap.parse_args()

    if args.smoke:
        args.model_name = "sshleifer/tiny-distilroberta-base"
        args.epochs, args.batch_size, args.max_len = 1, 8, 64

    device = pick_device(args.device)
    print(f"Device: {device} | model: {args.model_name}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    pool = load_pool(args.data_dir)
    if args.smoke:
        pool = pool.groupby("is_misogynistic", group_keys=False).head(100)
    n_pos = int(pool["is_misogynistic"].sum())
    n_neg = len(pool) - n_pos
    pos_weight = n_neg / max(1, n_pos)
    print(f"Pool: {len(pool):,} ({n_pos:,} pos / {n_neg:,} neg) | pos_weight={pos_weight:.2f}")

    train_ds = EvaluatorDataset(pool, tokenizer, args.max_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    biasly_test = pd.read_csv(args.data_dir / "biasly_test.csv")
    if args.smoke:
        biasly_test = biasly_test.head(60)
    id_loader = DataLoader(EvaluatorDataset(biasly_test, tokenizer, args.max_len),
                           batch_size=args.batch_size)
    selfma_ds = load_selfma(args.data_dir, tokenizer, args.max_len)
    ood_loader = DataLoader(selfma_ds, batch_size=args.batch_size) if selfma_ds else None

    model = MultiHeadEvaluator(model_name=args.model_name, binary_pos_weight=pos_weight).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_loader) * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * total_steps), total_steps)

    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for step, batch in enumerate(train_loader):
            opt.zero_grad()
            out = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                binary_label=batch["binary_label"].to(device),
                severity_label=batch["severity_label"].to(device),
                category_label=batch["category_label"].to(device),
                category_mask=batch["category_mask"].to(device),
            )
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            running += out.loss.item()
            if step % 50 == 0:
                print(f"  epoch {epoch+1} step {step}/{len(train_loader)} loss {out.loss.item():.4f}")
        print(f"epoch {epoch+1}: mean train loss {running/len(train_loader):.4f}")

    print("\n=== IN-DISTRIBUTION (Biasly held-out test) ===")
    id_metrics = evaluate(model, id_loader, device, with_category=True)
    print(json.dumps(id_metrics, indent=2))

    ood_metrics = {}
    if ood_loader:
        print("\n=== OUT-OF-DISTRIBUTION (SELFMA — real workplace dialogue, positives only) ===")
        ood_metrics = evaluate(model, ood_loader, device, with_category=False)
        # SELFMA is all-positive: recall is the catch-rate, the meaningful number.
        print(f"  catch-rate (recall on real MAs): {ood_metrics['binary']['recall']}")
        print("  (precision/F1 are undefined on a positives-only set — recall is the claim)")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.smoke:
        torch.save(model.state_dict(), args.output_dir / "evaluator.pt")
        tokenizer.save_pretrained(args.output_dir)
    (args.output_dir / "metrics.json").write_text(json.dumps(
        {"in_distribution": id_metrics, "ood_selfma": ood_metrics,
         "pool_size": len(pool), "pos_weight": pos_weight}, indent=2))
    print(f"\nSaved to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

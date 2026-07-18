"""
Pre-download a Hugging Face model + tokenizer into the local HF cache.

WHY THIS SCRIPT EXISTS
----------------------
AIRE compute nodes typically have no outbound internet, but training needs the
model weights. So the model must be fetched *once*, on a login node (which does
have internet), into a persistent cache on scratch. The training job then runs
fully offline against that cache.

This is a committed script rather than an ad-hoc terminal command so the step is
reproducible and visible in the project history.

USAGE (on an AIRE login node)
-----------------------------
    export HF_HOME=$SCRATCH/hf_cache          # or set in ~/.bashrc
    python scripts/predownload_model.py microsoft/deberta-v3-base

The same script serves the injector later (a BART/T5 model) — just pass its name.

It verifies the download by re-loading from cache with the hub disabled, i.e. it
reproduces exactly what the offline compute node will do, so a success here means
the job will not fail on a missing model.
"""
from __future__ import annotations

import argparse
import os
import sys


def main() -> int:
    ap = argparse.ArgumentParser(description="Pre-download a HF model into the cache")
    ap.add_argument("model", nargs="?", default="microsoft/deberta-v3-base",
                    help="HF model id (default: the evaluator's DeBERTa-v3-base)")
    args = ap.parse_args()

    hf_home = os.environ.get("HF_HOME")
    print(f"Model:   {args.model}")
    print(f"HF_HOME: {hf_home or '(unset — will use the default ~/.cache/huggingface)'}")
    if not hf_home:
        print("  WARNING: HF_HOME is not set. On AIRE, set it to a scratch path so the\n"
              "  cache persists and the compute node can find it:\n"
              "    export HF_HOME=$SCRATCH/hf_cache")

    # 1. Download (needs internet — run on a login node).
    from transformers import AutoModel, AutoTokenizer
    print("\nDownloading tokenizer...")
    AutoTokenizer.from_pretrained(args.model)
    print("Downloading model...")
    AutoModel.from_pretrained(args.model)
    print("  download complete.")

    # 2. Verify it loads OFFLINE — reproduces the compute node's conditions exactly.
    print("\nVerifying offline load (simulating the compute node)...")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    # Re-import against the offline flags.
    from importlib import reload
    import transformers
    reload(transformers)
    try:
        transformers.AutoTokenizer.from_pretrained(args.model)
        transformers.AutoModel.from_pretrained(args.model)
    except Exception as exc:
        print(f"  OFFLINE LOAD FAILED: {exc}")
        print("  The cache is incomplete — do not submit the training job yet.")
        return 1

    print("  offline load OK — the training job will find this model.")
    print(f"\nDone. '{args.model}' is cached and verified for offline use.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

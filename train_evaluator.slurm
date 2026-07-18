#!/bin/bash
#SBATCH --job-name=vawg_evaluator
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

# Train the misogyny evaluator (Objective C) on AIRE.
#
# This is MUCH simpler than run_pipeline.slurm: there is no Ollama server and no
# Apptainer container. The evaluator loads DeBERTa directly with transformers, so
# the job is just conda Python on a GPU.
#
# ── ONE-TIME SETUP, on a LOGIN node (which has internet; compute nodes may not) ──
#
#   1. Install the training dependencies into the conda env:
#        conda activate fyp
#        pip install -r requirements.txt
#
#   2. Pre-download the model to a persistent cache on scratch. Compute nodes are
#      commonly offline, so the model MUST be cached before the job runs. Use the
#      committed helper (it also verifies the OFFLINE load, reproducing the compute
#      node's conditions):
#        export HF_HOME=${SCRATCH}/hf_cache
#        python scripts/predownload_model.py microsoft/deberta-v3-base
#
#   The prepared data (data/evaluator/*.csv) is tracked in git, so it arrives with
#   the repo — no separate transfer needed. Only the RAW datasets (Individual
#   Project/*) are not in git, and they are only needed to RE-RUN preprocessing.

module load miniforge

# Use the env's python directly (matches run_pipeline.slurm) rather than
# `conda activate`, which is unreliable inside a non-interactive SLURM shell.
PYTHON=${HOME}/.conda/envs/fyp/bin/python

mkdir -p logs models

RUN_ID="$(date +%d%m%y_%H%M)_${SLURM_JOB_ID}"

# Use the model cache populated on the login node, and force offline mode so a
# compute node with no internet fails loudly here rather than hanging on a download.
export HF_HOME=${SCRATCH}/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# Reduce tokenizer thread contention warnings under SLURM cpus-per-task.
export TOKENIZERS_PARALLELISM=false

echo "── Config ──"
echo "  run:       ${RUN_ID}"
echo "  HF cache:  ${HF_HOME}"
${PYTHON} -c "import torch; print('  torch cuda:', torch.cuda.is_available(), '| device count:', torch.cuda.device_count())"

# Pre-flight: verify the cached model + tokenizer actually load offline BEFORE
# committing GPU time. Catches a missing cache (the most likely failure) in seconds.
echo "── Pre-flight: model cache ──"
${PYTHON} -c "
from transformers import AutoModel, AutoTokenizer
AutoTokenizer.from_pretrained('microsoft/deberta-v3-base')
AutoModel.from_pretrained('microsoft/deberta-v3-base')
print('  model + tokenizer load from cache OK')
" || {
    echo "FAILED: model not in cache. Run the pre-download step (see header) on a login node."
    exit 1
}

# ── Train ─────────────────────────────────────────────────────────────────────
${PYTHON} -m synthetic_conversation_generation.evaluator.train_evaluator \
    --data-dir data/evaluator \
    --output-dir models/evaluator_${RUN_ID} \
    --model-name microsoft/deberta-v3-base \
    --epochs 3 \
    --batch-size 16 \
    --lr 2e-5 \
    --device cuda

EXIT_CODE=$?

mv logs/${SLURM_JOB_ID}.out logs/eval_${RUN_ID}.out 2>/dev/null || true
mv logs/${SLURM_JOB_ID}.err logs/eval_${RUN_ID}.err 2>/dev/null || true

if [ $EXIT_CODE -eq 0 ]; then
    bash scripts/notify.sh end "evaluator_${RUN_ID}"   # metrics are in the log; notify summary is conversation-specific
    echo "Done. Model + metrics in models/evaluator_${RUN_ID}/"
    echo "── Metrics ──"
    cat models/evaluator_${RUN_ID}/metrics.json
else
    bash scripts/notify.sh fail "evaluator_${RUN_ID}" "logs/eval_${RUN_ID}.err"
    echo "Training failed with exit code ${EXIT_CODE}"
    exit ${EXIT_CODE}
fi

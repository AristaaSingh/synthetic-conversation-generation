#!/bin/bash
#SBATCH --job-name=vawg_pipeline
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

# ── Environment ───────────────────────────────────────────────────────────────
module load apptainer
module load miniforge

PYTHON=${HOME}/.conda/envs/fyp/bin/python
NOTIFY=scripts/notify.sh

# ── Timestamp and run ID ─────────────────────────────────────────────────────
TIMESTAMP=$(date "+%d%m%y_%H%M")
RUN_ID="${TIMESTAMP}_${SLURM_JOB_ID}"

# ── Paths ─────────────────────────────────────────────────────────────────────
OLLAMA_SIF=${SCRATCH}/ollama.sif
OLLAMA_MODELS=${SCRATCH}/models
CONVERSATIONS_DIR=${SCRATCH}/conversations

# Only create subdirs we own — never attempt to create $SCRATCH itself
mkdir -p ${OLLAMA_MODELS} ${CONVERSATIONS_DIR} logs

bash ${NOTIFY} start ${SLURM_JOB_ID}

# ── Start Ollama server inside the container ──────────────────────────────────
export OLLAMA_HOST=127.0.0.1:11434
export OLLAMA_MODELS=${OLLAMA_MODELS}

apptainer exec --nv \
    --bind ${SCRATCH}:${SCRATCH} \
    --env OLLAMA_HOST=${OLLAMA_HOST} \
    --env OLLAMA_MODELS=${OLLAMA_MODELS} \
    ${OLLAMA_SIF} ollama serve &
OLLAMA_PID=$!

echo "Waiting for Ollama to start..."
sleep 15

# ── Pull model if not already cached ─────────────────────────────────────────
apptainer exec --nv \
    --bind ${SCRATCH}:${SCRATCH} \
    --env OLLAMA_HOST=${OLLAMA_HOST} \
    --env OLLAMA_MODELS=${OLLAMA_MODELS} \
    ${OLLAMA_SIF} ollama pull gpt-oss:20b

# ── Run the pipeline ──────────────────────────────────────────────────────────
${PYTHON} -m synthetic_conversation_generation.pipeline \
    --character-a data/characters/victims/sophie_walker.yaml \
    --character-b data/characters/perpetrators/ryan_chambers.yaml \
    --world data/worlds/uk_tech_company.yaml \
    --model-provider ollama \
    --model-id gpt-oss:20b \
    --max-turns 60 \
    --max-sessions 6 \
    --output-path ${CONVERSATIONS_DIR}/${RUN_ID}.json \
    --conversation-id ${RUN_ID}

EXIT_CODE=$?

# ── Cleanup ───────────────────────────────────────────────────────────────────
kill ${OLLAMA_PID}

# Rename log files from jobID-only to DDMMYY_HHMM_jobID format
mv logs/${SLURM_JOB_ID}.out logs/${RUN_ID}.out 2>/dev/null || true
mv logs/${SLURM_JOB_ID}.err logs/${RUN_ID}.err 2>/dev/null || true

if [ $EXIT_CODE -eq 0 ]; then
    bash ${NOTIFY} end ${SLURM_JOB_ID}
    echo "Done. Output saved to ${CONVERSATIONS_DIR}/${RUN_ID}.json"
else
    bash ${NOTIFY} fail ${SLURM_JOB_ID} logs/${RUN_ID}.err
    echo "Pipeline failed with exit code ${EXIT_CODE}"
    exit ${EXIT_CODE}
fi

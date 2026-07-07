#!/bin/bash
#SBATCH --job-name=vawg_pipeline
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# ── Environment ───────────────────────────────────────────────────────────────
module load apptainer
module load miniforge

PYTHON=${HOME}/.conda/envs/fyp/bin/python
NOTIFY=${HOME}/synthetic-conversation-generation/scripts/notify.sh

# ── Paths ─────────────────────────────────────────────────────────────────────
CODE_DIR=${HOME}/synthetic-conversation-generation
OLLAMA_SIF=${SCRATCH}/ollama.sif
OLLAMA_MODELS=${SCRATCH}/models
CONVERSATIONS_DIR=${SCRATCH}/conversations
ERROR_LOG=${CODE_DIR}/logs/vawg_pipeline_${SLURM_JOB_ID}.err

mkdir -p ${OLLAMA_MODELS} ${CONVERSATIONS_DIR}

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
cd ${CODE_DIR}
${PYTHON} -m synthetic_conversation_generation.pipeline \
    --character-a data/characters/priya_sharma.yaml \
    --character-b data/characters/james_whitmore.yaml \
    --scenario data/scenarios/microaggression_sexism.yaml \
    --model-provider ollama \
    --model-id gpt-oss:20b \
    --max-turns 10 \
    --output-path ${CONVERSATIONS_DIR}/run_${SLURM_JOB_ID}.json \
    --conversation-id ${SLURM_JOB_ID}

EXIT_CODE=$?

# ── Cleanup ───────────────────────────────────────────────────────────────────
kill ${OLLAMA_PID}

if [ $EXIT_CODE -eq 0 ]; then
    bash ${NOTIFY} end ${SLURM_JOB_ID}
    echo "Done. Output saved to ${CONVERSATIONS_DIR}/run_${SLURM_JOB_ID}.json"
else
    bash ${NOTIFY} fail ${SLURM_JOB_ID} ${ERROR_LOG}
    echo "Pipeline failed with exit code ${EXIT_CODE}"
    exit ${EXIT_CODE}
fi

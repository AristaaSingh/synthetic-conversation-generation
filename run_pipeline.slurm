#!/bin/bash
#SBATCH --job-name=vawg_pipeline
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=aristaasingh@gmail.com

# ── Environment ───────────────────────────────────────────────────────────────
module load apptainer
module load miniforge

# Use Python directly from the conda env — avoids conda activate issues in SLURM
PYTHON=${HOME}/.conda/envs/fyp/bin/python

# ── Paths ─────────────────────────────────────────────────────────────────────
CODE_DIR=${HOME}/synthetic-conversation-generation
OLLAMA_SIF=${SCRATCH}/ollama.sif
OLLAMA_MODELS=${SCRATCH}/models
CONVERSATIONS_DIR=${SCRATCH}/conversations

mkdir -p ${OLLAMA_MODELS} ${CONVERSATIONS_DIR}

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

# ── Cleanup ───────────────────────────────────────────────────────────────────
kill ${OLLAMA_PID}
echo "Done. Output saved to ${CONVERSATIONS_DIR}/run_${SLURM_JOB_ID}.json"

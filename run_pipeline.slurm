#!/bin/bash
#SBATCH --job-name=vawg_pipeline
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

# ── Environment ───────────────────────────────────────────────────────────────
module load apptainer
module load miniforge

PYTHON=${HOME}/.conda/envs/fyp/bin/python
NOTIFY=scripts/notify.sh

# ── Run configuration ────────────────────────────────────────────────────────
# TARGET_DAYS is the termination condition: the run ends when the SIMULATED clock
# passes it. Turn count is emergent — a fortnight spent in escalation produces far
# more messages than one spent in post_incident, which is the realistic behaviour.
# Sessions are unbounded.
#
# MAX_TURNS is a CIRCUIT BREAKER against a runaway run eating the wall clock, not a
# design parameter. If it fires, investigate: the run did not finish normally.
# See project_record.md 22, 23.8 and 25.
TARGET_DAYS=14
MAX_TURNS=300
EXCHANGE_BUDGET=5
MODEL=gpt-oss:20b
CHARACTER_A=data/characters/victims/sophie_walker.yaml
CHARACTER_B=data/characters/perpetrators/ryan_chambers.yaml
WORLD=data/worlds/uk_tech_company.yaml

# ── Timestamp and run ID ─────────────────────────────────────────────────────
TIMESTAMP=$(date "+%d%m%y_%H%M")
RUN_ID="${TIMESTAMP}_${SLURM_JOB_ID}"

# ── Paths ─────────────────────────────────────────────────────────────────────
OLLAMA_SIF=${SCRATCH}/ollama.sif
OLLAMA_MODELS=${SCRATCH}/models
CONVERSATIONS_DIR=${SCRATCH}/conversations

# Only create subdirs we own — never attempt to create $SCRATCH itself
mkdir -p ${OLLAMA_MODELS} ${CONVERSATIONS_DIR} logs

bash ${NOTIFY} start ${RUN_ID}

# ── Start Ollama server inside the container ──────────────────────────────────
export OLLAMA_HOST=127.0.0.1:11434
export OLLAMA_MODELS=${OLLAMA_MODELS}

apptainer exec --nv \
    --bind ${SCRATCH}:${SCRATCH} \
    --env OLLAMA_HOST=${OLLAMA_HOST} \
    --env OLLAMA_MODELS=${OLLAMA_MODELS} \
    ${OLLAMA_SIF} ollama serve &
OLLAMA_PID=$!

# Poll for readiness rather than sleeping a fixed 15s. A fixed sleep is a guess:
# too short and the first query fails, too long and it wastes wall clock.
echo "Waiting for Ollama to become ready..."
for i in $(seq 1 60); do
    if curl -sf "http://${OLLAMA_HOST}/api/tags" >/dev/null 2>&1; then
        echo "Ollama ready after ${i}s"
        break
    fi
    sleep 1
done

# ── Pull model if not already cached ─────────────────────────────────────────
apptainer exec --nv \
    --bind ${SCRATCH}:${SCRATCH} \
    --env OLLAMA_HOST=${OLLAMA_HOST} \
    --env OLLAMA_MODELS=${OLLAMA_MODELS} \
    ${OLLAMA_SIF} ollama pull ${MODEL}

# ── Pre-flight ────────────────────────────────────────────────────────────────
# Control-flow tests run without an LLM in ~1s. Catching a broken config here
# costs seconds; catching it after 40 minutes of GPU inference does not.
echo "── Pre-flight tests ──"
${PYTHON} -m synthetic_conversation_generation.test_pipeline || {
    echo "Pre-flight tests FAILED — aborting before burning GPU time."
    kill ${OLLAMA_PID}
    bash ${NOTIFY} fail "${RUN_ID}" "logs/${SLURM_JOB_ID}.err"
    exit 1
}

OUTPUT_FILE=${CONVERSATIONS_DIR}/${RUN_ID}.json

echo "── Run config ──"
echo "  model:      ${MODEL}"
echo "  target:     ${TARGET_DAYS} simulated days (turns emergent; breaker at ${MAX_TURNS})"
echo "  characters: $(basename ${CHARACTER_A}) <-> $(basename ${CHARACTER_B})"
echo "  world:      $(basename ${WORLD})"
echo "  output:     ${OUTPUT_FILE}"

# ── Run the pipeline ──────────────────────────────────────────────────────────
# The pipeline checkpoints to --output-path after every exchange and writes
# atomically, so if this job hits the SLURM wall clock the partial conversation
# survives with "complete": false. Check that field before using any output.
${PYTHON} -m synthetic_conversation_generation.pipeline \
    --character-a ${CHARACTER_A} \
    --character-b ${CHARACTER_B} \
    --world ${WORLD} \
    --model-provider ollama \
    --model-id ${MODEL} \
    --target-days ${TARGET_DAYS} \
    --max-turns ${MAX_TURNS} \
    --exchange-budget ${EXCHANGE_BUDGET} \
    --output-path ${OUTPUT_FILE} \
    --conversation-id ${RUN_ID}

EXIT_CODE=$?

# ── Cleanup ───────────────────────────────────────────────────────────────────
kill ${OLLAMA_PID}

# Rename log files from jobID-only to DDMMYY_HHMM_jobID format
mv logs/${SLURM_JOB_ID}.out logs/${RUN_ID}.out 2>/dev/null || true
mv logs/${SLURM_JOB_ID}.err logs/${RUN_ID}.err 2>/dev/null || true

if [ $EXIT_CODE -eq 0 ]; then
    bash ${NOTIFY} end "${RUN_ID}" "${OUTPUT_FILE}"
    echo "Done. Output saved to ${OUTPUT_FILE}"
    echo "── Run summary ──"
    ${PYTHON} -m synthetic_conversation_generation.summarise_run "${OUTPUT_FILE}"
else
    bash ${NOTIFY} fail "${RUN_ID}" "logs/${RUN_ID}.err"
    echo "Pipeline failed with exit code ${EXIT_CODE}"
    # A partial checkpoint may still exist — a wall-clock timeout is the likely cause.
    if [ -f "${OUTPUT_FILE}" ]; then
        echo "A partial checkpoint survived at ${OUTPUT_FILE} (check \"complete\": false)"
    fi
    exit ${EXIT_CODE}
fi

#!/bin/bash
# Usage: notify.sh <start|end|fail> <run_id> [error_log_or_output_json]
#
# Sends a phone push (ntfy) and an email. Push is primary — email arrives too
# late to be useful for knowing when a job has finished.
#
# ── ntfy setup (one-off, ~1 minute) ───────────────────────────────────────────
#   1. Install the "ntfy" app (iOS / Android / F-Droid), or open https://ntfy.sh
#   2. Subscribe to the topic named in NTFY_TOPIC below
#   3. Test:  curl -d "hello" https://ntfy.sh/aristaa-fyp
#
# The topic name IS the only access control — anyone who knows it can read and
# post to it. Keep it unguessable, and never put conversation content in a
# notification: job IDs and metrics only.
#
# If AIRE compute nodes block outbound HTTPS, the push silently no-ops and the
# email still goes. Verify from a compute node with:
#   curl -sf -d "test" https://ntfy.sh/aristaa-fyp && echo OK
#
# A notification must never fail the job, so every network call ends in `|| true`.

set -uo pipefail

EVENT="${1:-}"
RUN_ID="${2:-unknown}"
EXTRA="${3:-}"

RECIPIENT="aristaasingh@gmail.com"

# Override per-shell with `export NTFY_TOPIC=...`; empty disables push entirely.
NTFY_TOPIC="${NTFY_TOPIC:-aristaa-fyp}"

TIMESTAMP=$(date "+%d/%m/%Y %H:%M:%S")

push() {
    # push <title> <priority 1-5> <tags> <body>
    [ -z "${NTFY_TOPIC}" ] && return 0
    curl -sf --max-time 10 \
        -H "Title: $1" \
        -H "Priority: $2" \
        -H "Tags: $3" \
        -d "$4" \
        "https://ntfy.sh/${NTFY_TOPIC}" >/dev/null 2>&1 || true
}

send_email() {
    # send_email <subject> <body>
    command -v mail >/dev/null 2>&1 || return 0
    echo "$2" | mail -s "$1" "${RECIPIENT}" 2>/dev/null || true
}

case "${EVENT}" in
  start)
    BODY="Job ${RUN_ID} started on AIRE.

Node:      ${SLURMD_NODENAME:-?}
Partition: ${SLURM_JOB_PARTITION:-?}
Started:   ${TIMESTAMP}

Logs: logs/${RUN_ID}.out / logs/${RUN_ID}.err"
    # Priority 2 (low) — a start is informational; it should not buzz.
    push "AIRE: run started" 2 "hourglass" "${RUN_ID} on ${SLURMD_NODENAME:-node}"
    send_email "AIRE: run ${RUN_ID} started" "${BODY}"
    ;;

  end)
    # $EXTRA is the output JSON when supplied. Put the headline metrics in the push
    # itself, so the result is readable from a lock screen without opening a laptop.
    SUMMARY=""
    if [ -n "${EXTRA}" ] && [ -f "${EXTRA}" ]; then
        SUMMARY=$(python3 - "${EXTRA}" 2>/dev/null <<'PYEOF' || true
import json, sys
from datetime import datetime
d = json.load(open(sys.argv[1]))
ts = [datetime.strptime(m["timestamp"], "%Y-%m-%d %H:%M") for m in d["messages"]]
span = (ts[-1] - ts[0]).total_seconds() / 86400 if len(ts) > 1 else 0
fs = d["final_state"]
flag = "" if d.get("complete") else "  [INCOMPLETE]"
print(f"{d['turns_generated']} turns / {len(d['dialogue_flows'])} sessions{flag}")
print(f"span {span:.1f}d | tension {fs['tension_level']}/5 | {fs['phase']}")
print(f"incident: {fs['incident_occurred']}")
PYEOF
)
    fi
    [ -z "${SUMMARY}" ] && SUMMARY="(no summary available)"

    BODY="Job ${RUN_ID} completed.

Finished: ${TIMESTAMP}

${SUMMARY}

Output: ${EXTRA:-\$SCRATCH/conversations/${RUN_ID}.json}"
    # Priority 4 (high) — breaks through Do Not Disturb.
    push "AIRE: run finished" 4 "white_check_mark" "${RUN_ID}
${SUMMARY}"
    send_email "AIRE: run ${RUN_ID} finished" "${BODY}"
    ;;

  fail)
    ERROR_CONTENT=""
    if [ -n "${EXTRA}" ] && [ -f "${EXTRA}" ]; then
        # Ollama logs everything to stderr at INFO level, so a plain tail is mostly
        # noise. Pull the lines that actually indicate a failure; fall back to a tail.
        ERROR_CONTENT=$(grep -iE "traceback|error|exception|failed" "${EXTRA}" 2>/dev/null | tail -15)
        [ -z "${ERROR_CONTENT}" ] && ERROR_CONTENT=$(tail -20 "${EXTRA}" 2>/dev/null)
    fi
    LAST_LINE=$(echo "${ERROR_CONTENT}" | tail -1)

    BODY="Job ${RUN_ID} FAILED.

Failed at: ${TIMESTAMP}

Filtered from the error log:
---
${ERROR_CONTENT}
---

A partial checkpoint may have survived — check \"complete\": false in
\$SCRATCH/conversations/${RUN_ID}.json"
    # Priority 5 (urgent) — a failed run wastes the queue slot until noticed.
    push "AIRE: run FAILED" 5 "rotating_light" "${RUN_ID}
${LAST_LINE}"
    send_email "AIRE: run ${RUN_ID} FAILED" "${BODY}"
    ;;

  *)
    echo "notify.sh: unknown event '${EVENT}' (expected start|end|fail)" >&2
    exit 1
    ;;
esac

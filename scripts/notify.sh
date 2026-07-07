#!/bin/bash
# Usage: notify.sh <start|end|fail> <job_id> [error_log_path]

EVENT=$1
JOB_ID=$2
ERROR_LOG=$3

RECIPIENT="aristaasingh@gmail.com"
TIMESTAMP=$(date "+%d/%m/%Y %H:%M:%S")
SUBJECT="Slurm job at ${TIMESTAMP}"

case $EVENT in
  start)
    BODY="Job ${JOB_ID} has started running on AIRE.

Job name:   vawg_pipeline
Job ID:     ${JOB_ID}
Started at: ${TIMESTAMP}
Node:       ${SLURMD_NODENAME}
Partition:  ${SLURM_JOB_PARTITION}

Logs will be written to:
  logs/vawg_pipeline_${JOB_ID}.out
  logs/vawg_pipeline_${JOB_ID}.err"
    ;;

  end)
    BODY="Job ${JOB_ID} completed successfully.

Job name:    vawg_pipeline
Job ID:      ${JOB_ID}
Finished at: ${TIMESTAMP}
Output:      \$SCRATCH/conversations/run_${JOB_ID}.json"
    ;;

  fail)
    ERROR_CONTENT=""
    if [ -f "$ERROR_LOG" ]; then
      ERROR_CONTENT=$(tail -30 "$ERROR_LOG")
    fi
    BODY="Job ${JOB_ID} failed.

Job name:  vawg_pipeline
Job ID:    ${JOB_ID}
Failed at: ${TIMESTAMP}

Last 30 lines of error log:
---
${ERROR_CONTENT}"
    ;;
esac

echo "$BODY" | mail -s "$SUBJECT" "$RECIPIENT"

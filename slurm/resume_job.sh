#!/bin/bash

fd_run_requeueable() {
  FD_CHILD_PID=""

  fd_requeue() {
    kill -TERM "${FD_CHILD_PID}" 2>/dev/null || true
    if (( ${SLURM_RESTART_COUNT:-0} >= 6 )); then
      echo "Resume attempt cap reached." >&2
      exit 124
    fi
    local target="${SLURM_JOB_ID}"
    if [[ -n "${SLURM_ARRAY_JOB_ID:-}" && -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
      target="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
    fi
    echo "Time limit approaching; requeuing ${target}."
    if ! scontrol requeue "${target}"; then
      echo "Failed to requeue ${target}." >&2
      exit 1
    fi
    exit 0
  }
  trap fd_requeue USR1

  set +e
  "$@" &
  FD_CHILD_PID=$!
  wait "${FD_CHILD_PID}"
  FD_STATUS=$?
  set -e
  return "${FD_STATUS}"
}

#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PARENT_DIR="$(cd "${BASE_DIR}/.." && pwd)"

VENV_DIR="${VENV_DIR:-${PARENT_DIR}/fl_venv}"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/logs}"
CONTAINER="${CONTAINER:-/ceph/container/pytorch/pytorch_25.08.sif}"

NUM_CLIENTS="$1"
DATASET_NAME="$2"
EXCLUDE_NODE="${3:-}"

MAX_SUPERNODES_PER_JOB="${MAX_SUPERNODES_PER_JOB:-4}"
MAX_RETRIES="${MAX_RETRIES:-2}"

mkdir -p "$LOG_DIR"

print_section() {
  echo "========================================"
  echo "$1"
  echo "========================================"
}

wait_for_bundle_or_retry() {
  local job_id="$1"
  local label="$2"

  while true; do
    state="$(squeue -j "$job_id" -h -o "%T" || true)"

    if [ -z "$state" ]; then
      final_state="$(sacct -j "$job_id" -n -X --format=State | head -n 1 | awk '{print $1}' || true)"
      echo "${label} finished with state: ${final_state}"

      case "$final_state" in
        COMPLETED|RUNNING)
          return 0
          ;;
        *)
          return 1
          ;;
      esac
    fi

    if [ "$state" = "RUNNING" ]; then
      echo "${label} is RUNNING."
      return 0
    fi

    echo "  ${label} state: ${state}"
    sleep 10
  done
}

submit_one_bundle() {
  local start="$1"
  local end="$2"

  SBATCH_ARGS=()
  if [ -n "$EXCLUDE_NODE" ]; then
    SBATCH_ARGS+=(--exclude="$EXCLUDE_NODE")
  fi

  sbatch --parsable \
    "${SBATCH_ARGS[@]}" \
    --export=ALL,SUPERLINK_HOST="${SUPERLINK_HOST}",CLIENT_START="${start}",CLIENT_END="${end}",NUM_CLIENTS="${NUM_CLIENTS}",DATASET_NAME="${DATASET_NAME}",BASE_DIR="${BASE_DIR}",VENV_DIR="${VENV_DIR}",LOG_DIR="${LOG_DIR}",CONTAINER="${CONTAINER}" \
    --output="${LOG_DIR}/supernodes-%j.out" \
    --error="${LOG_DIR}/supernodes-%j.err" \
    "${SCRIPT_DIR}/supernodes.sbatch"
}

submit_supernode_bundles_with_retry() {
  print_section "Submitting SuperNode bundles"

  START=1

  while [ "$START" -le "$NUM_CLIENTS" ]; do
    END=$((START + MAX_SUPERNODES_PER_JOB - 1))
    [ "$END" -gt "$NUM_CLIENTS" ] && END="$NUM_CLIENTS"

    ATTEMPT=1
    SUCCESS=0

    while [ "$ATTEMPT" -le "$((MAX_RETRIES + 1))" ]; do
      echo "Bundle ${START}-${END} attempt ${ATTEMPT}"

      JOB_ID=$(submit_one_bundle "$START" "$END")
      echo "  -> Job ID: ${JOB_ID}"

      if wait_for_bundle_or_retry "$JOB_ID" "Bundle ${START}-${END}"; then
        SUCCESS=1
        break
      fi

      echo "Retrying bundle ${START}-${END}..."
      ATTEMPT=$((ATTEMPT + 1))
      sleep 10
    done

    if [ "$SUCCESS" -ne 1 ]; then
      echo "Bundle ${START}-${END} failed after retries"
      exit 1
    fi

    START=$((END + 1))
  done
}

submit_superlink() {
  print_section "Starting SuperLink"

  SUPERLINK_JOB_ID=$(sbatch --parsable \
    --export=ALL,BASE_DIR="${BASE_DIR}",VENV_DIR="${VENV_DIR}",LOG_DIR="${LOG_DIR}",CONTAINER="${CONTAINER}" \
    --output="${LOG_DIR}/superlink-%j.out" \
    --error="${LOG_DIR}/superlink-%j.err" \
    "${SCRIPT_DIR}/superlink.sbatch")

  echo "SuperLink Job: $SUPERLINK_JOB_ID"

  for i in {1..60}; do
    if grep -q "Starting SuperLink on host:" "${LOG_DIR}/superlink-${SUPERLINK_JOB_ID}.out" 2>/dev/null; then
      SUPERLINK_HOST=$(grep "Starting SuperLink on host:" "${LOG_DIR}/superlink-${SUPERLINK_JOB_ID}.out" | tail -1 | awk '{print $5}')
      export SUPERLINK_HOST
      echo "SuperLink host: $SUPERLINK_HOST"
      return
    fi
    sleep 2
  done

  echo "Failed to get SuperLink host"
  exit 1
}

submit_serverapp() {
  print_section "Starting ServerApp"

  sbatch \
    --export=ALL,SUPERLINK_HOST="${SUPERLINK_HOST}",BASE_DIR="${BASE_DIR}",VENV_DIR="${VENV_DIR}",LOG_DIR="${LOG_DIR}",CONTAINER="${CONTAINER}" \
    --output="${LOG_DIR}/serverapp-%j.out" \
    --error="${LOG_DIR}/serverapp-%j.err" \
    "${SCRIPT_DIR}/serverapp.sbatch"
}

print_section "Federated run"
echo "Clients: $NUM_CLIENTS"
echo "Dataset: $DATASET_NAME"

submit_superlink
submit_supernode_bundles_with_retry
submit_serverapp

print_section "Done"
#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PARENT_DIR="$(cd "${BASE_DIR}/.." && pwd)"
VENV_DIR="${VENV_DIR:-${PARENT_DIR}/fl_venv}"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/logs}"
FLWR_HOME="${FLWR_HOME:-${HOME}/.flwr}"

export BASE_DIR
export VENV_DIR
export LOG_DIR
export FLWR_HOME
export SCRIPT_DIR

SERVERAPP_JOB_ID=""
SUPERLINK_JOB_ID=""
SUPERNODE_JOB_IDS=()

cd "$BASE_DIR"
mkdir -p "$LOG_DIR"

if [ $# -lt 2 ]; then
  echo "Usage: bash ${SCRIPT_DIR}/launch_all.sh <NUM_CLIENTS> <DATASET_NAME>"
  exit 1
fi

NUM_CLIENTS="$1"
DATASET_NAME="$2"

export NUM_CLIENTS
export DATASET_NAME

if ! [[ "$NUM_CLIENTS" =~ ^[0-9]+$ ]] || [ "$NUM_CLIENTS" -lt 1 ]; then
  echo "NUM_CLIENTS must be a positive integer."
  exit 1
fi

print_section() {
  echo "========================================"
  echo "$1"
  echo "========================================"
}

cleanup_old_jobs() {
  print_section "Cleaning up old Flower jobs..."
  scancel -u "$USER" -n fl-superlink 2>/dev/null || true
  scancel -u "$USER" -n fl-supernode 2>/dev/null || true
  scancel -u "$USER" -n fl-serverapp 2>/dev/null || true
  echo "Waiting a few seconds for ports/resources to release..."
  sleep 5
}

cleanup_logs() {
  print_section "Cleaning old log files..."
  rm -f "${LOG_DIR}"/superlink-*.out "${LOG_DIR}"/superlink-*.err \
        "${LOG_DIR}"/supernode-*.out "${LOG_DIR}"/supernode-*.err \
        "${LOG_DIR}"/serverapp-*.out "${LOG_DIR}"/serverapp-*.err
}

SUCCESS=0

cleanup_on_exit() {
  if [ "$SUCCESS" -eq 1 ]; then
    return
  fi

  echo
  print_section "Cleaning up Flower jobs after interruption/failure..."
  scancel -u "$USER" -n fl-superlink 2>/dev/null || true
  scancel -u "$USER" -n fl-supernode 2>/dev/null || true
  scancel -u "$USER" -n fl-serverapp 2>/dev/null || true
}

trap cleanup_on_exit EXIT INT TERM

check_datasets() {
  print_section "Checking dataset files for dataset: ${DATASET_NAME}"

  for ((i=1; i<=NUM_CLIENTS; i++)); do
    DATASET_PATH="${BASE_DIR}/datasets/${DATASET_NAME}/data${i}.csv"
    if [ ! -f "$DATASET_PATH" ]; then
      echo "Missing dataset file: $DATASET_PATH"
      exit 1
    fi
    echo "Found: $DATASET_PATH"
  done
}

update_num_clients_in_config() {
  print_section "Updating config.py num_clients -> ${NUM_CLIENTS}"

  python3 - "$BASE_DIR/config.py" "$NUM_CLIENTS" <<'PY'
import re
import sys
from pathlib import Path

config_path = Path(sys.argv[1])
num_clients = sys.argv[2]

text = config_path.read_text(encoding="utf-8")
pattern = r'(^\s*num_clients\s*:\s*int\s*=\s*)\d+'
new_text, count = re.subn(pattern, rf'\g<1>{num_clients}', text, flags=re.MULTILINE)

if count == 0:
    print("Failed to update num_clients in config.py")
    sys.exit(1)

config_path.write_text(new_text, encoding="utf-8")
print("Updated num_clients successfully.")
PY
}

wait_for_job_running() {
  local job_id="$1"
  local label="$2"

  echo "Waiting for ${label} job to start running..."

  for _ in {1..300}; do
    JOB_STATE="$(squeue -j "$job_id" -h -o "%T" || true)"

    if [ "$JOB_STATE" = "RUNNING" ]; then
      echo "${label} job is RUNNING."
      return 0
    fi

    if [ -z "$JOB_STATE" ]; then
      echo "${label} job disappeared before reaching RUNNING."
      sacct -j "$job_id" --format=JobID,State,ExitCode || true
      exit 1
    fi

    sleep 2
  done


  echo "Timed out waiting for ${label} job to start."
  squeue -j "$job_id" -o "%.18i %.9P %.20j %.8u %.2t %.10M %.6D %R" || true
  scontrol show job "$job_id" || true
  sacct -j "$job_id" --format=JobID,State,ExitCode || true
  exit 1
}

wait_for_superlink_host() {
  echo "Waiting for SuperLink host to appear in log..."

  for _ in {1..60}; do
    if [ -f "$SUPERLINK_LOG" ] && grep -q "Starting SuperLink on host:" "$SUPERLINK_LOG"; then
      SUPERLINK_HOST="$(grep "Starting SuperLink on host:" "$SUPERLINK_LOG" | tail -n 1 | awk '{print $5}')"
      if [ -n "${SUPERLINK_HOST:-}" ]; then
        echo "SuperLink host detected: $SUPERLINK_HOST"
        return 0
      fi
    fi
    sleep 2
  done

  echo "Failed to determine SuperLink host from log."
  cat "$SUPERLINK_LOG" 2>/dev/null || true
  exit 1
}

wait_for_superlink_ready() {
  echo "Waiting for SuperLink API to become ready..."

  READY=0
  for _ in {1..60}; do
    if [ -f "$SUPERLINK_ERR_LOG" ] && grep -q "Starting Control API on 0.0.0.0:9093" "$SUPERLINK_ERR_LOG"; then
      READY=1
      break
    fi

    JOB_STATE="$(squeue -j "$SUPERLINK_JOB_ID" -h -o "%T" || true)"

    if [ -z "$JOB_STATE" ]; then
      echo "SuperLink job disappeared before becoming ready."
      sacct -j "$SUPERLINK_JOB_ID" --format=JobID,State,ExitCode || true
      echo "--- superlink stdout ---"
      cat "$SUPERLINK_LOG" 2>/dev/null || true
      echo "--- superlink stderr ---"
      cat "$SUPERLINK_ERR_LOG" 2>/dev/null || true
      exit 1
    fi

    sleep 2
  done

  if [ "$READY" -ne 1 ]; then
    echo "SuperLink API did not become ready in time."
    cat "$SUPERLINK_LOG" 2>/dev/null || true
    cat "$SUPERLINK_ERR_LOG" 2>/dev/null || true
    exit 1
  fi

  echo "SuperLink is ready."
}

submit_supernodes() {
  print_section "Submitting ${NUM_CLIENTS} SuperNode jobs for dataset: ${DATASET_NAME}"
  SUPERNODE_JOB_IDS=()

  for ((i=1; i<=NUM_CLIENTS; i++)); do
    FACILITY_ID="client-${i}"
    DATA_PATH="/workspace/fl_backend/datasets/${DATASET_NAME}/data${i}.csv"
    PARTITION_ID=$((i - 1))
    CLIENTAPPIO_PORT=$((9093 + i))

    JOB_ID=$(sbatch --parsable \
      --export=ALL,NUM_CLIENTS="${NUM_CLIENTS}",DATASET_NAME="${DATASET_NAME}",BASE_DIR="${BASE_DIR}",VENV_DIR="${VENV_DIR}",LOG_DIR="${LOG_DIR}",FLWR_HOME="${FLWR_HOME}" \
      --chdir="$BASE_DIR" \
      --output="${LOG_DIR}/supernode-%j.out" \
      --error="${LOG_DIR}/supernode-%j.err" \
      "${SCRIPT_DIR}/supernode.sbatch" \
      "$SUPERLINK_HOST" "$FACILITY_ID" "$DATA_PATH" "$PARTITION_ID" "$CLIENTAPPIO_PORT")

    echo "Submitted SuperNode ${FACILITY_ID} -> ${JOB_ID}"
    SUPERNODE_JOB_IDS+=("$JOB_ID")
  done
}

wait_for_supernodes_running() {
  print_section "Waiting for all SuperNode jobs to start..."

  for JOB_ID in "${SUPERNODE_JOB_IDS[@]}"; do
    echo "Waiting for SuperNode job ${JOB_ID}..."

    for _ in {1..120}; do
      JOB_STATE="$(squeue -j "$JOB_ID" -h -o "%T" || true)"

      if [ "$JOB_STATE" = "RUNNING" ]; then
        echo "SuperNode job ${JOB_ID} is RUNNING."
        break
      fi

      if [ -z "$JOB_STATE" ]; then
        echo "SuperNode job ${JOB_ID} disappeared before reaching RUNNING."
        sacct -j "$JOB_ID" --format=JobID,State,ExitCode,NodeList,Elapsed || true
        exit 1
      fi

      echo "  Current state: ${JOB_STATE}"
      sleep 5
    done
  done

  echo "All SuperNode jobs are RUNNING."
}

submit_serverapp() {
  print_section "Submitting ServerApp job..."

  SERVERAPP_JOB_ID=$(sbatch --parsable \
    --export=ALL,NUM_CLIENTS="${NUM_CLIENTS}",DATASET_NAME="${DATASET_NAME}",BASE_DIR="${BASE_DIR}",VENV_DIR="${VENV_DIR}",LOG_DIR="${LOG_DIR}",FLWR_HOME="${FLWR_HOME}" \
    --chdir="$BASE_DIR" \
    --output="${LOG_DIR}/serverapp-%j.out" \
    --error="${LOG_DIR}/serverapp-%j.err" \
    "${SCRIPT_DIR}/serverapp.sbatch" \
    "$SUPERLINK_HOST")

  echo "ServerApp job ID: $SERVERAPP_JOB_ID"
}

wait_for_serverapp_finish() {
  print_section "Waiting for ServerApp job to finish..."

  while true; do
    JOB_STATE="$(squeue -j "$SERVERAPP_JOB_ID" -h -o "%T" || true)"

    if [ -z "$JOB_STATE" ]; then
      echo "ServerApp job has left the queue."
      sacct -j "$SERVERAPP_JOB_ID" --format=JobID,State,ExitCode,Elapsed || true
      return 0
    fi

    echo "  ServerApp state: ${JOB_STATE}"
    sleep 10
  done
}

cleanup_old_jobs
cleanup_logs
check_datasets
update_num_clients_in_config

print_section "Submitting SuperLink job..."
SUPERLINK_JOB_ID=$(sbatch --parsable \
  --export=ALL,NUM_CLIENTS="${NUM_CLIENTS}",DATASET_NAME="${DATASET_NAME}",BASE_DIR="${BASE_DIR}",VENV_DIR="${VENV_DIR}",LOG_DIR="${LOG_DIR}",FLWR_HOME="${FLWR_HOME}" \
  --chdir="$BASE_DIR" \
  --output="${LOG_DIR}/superlink-%j.out" \
  --error="${LOG_DIR}/superlink-%j.err" \
  "${SCRIPT_DIR}/superlink.sbatch")

echo "SuperLink job ID: $SUPERLINK_JOB_ID"

SUPERLINK_LOG="${LOG_DIR}/superlink-${SUPERLINK_JOB_ID}.out"
SUPERLINK_ERR_LOG="${LOG_DIR}/superlink-${SUPERLINK_JOB_ID}.err"

wait_for_job_running "$SUPERLINK_JOB_ID" "SuperLink"
wait_for_superlink_host
wait_for_superlink_ready

submit_supernodes
wait_for_supernodes_running

echo "Waiting a few extra seconds before starting Flower run..."
sleep 10

submit_serverapp
wait_for_job_running "$SERVERAPP_JOB_ID" "ServerApp"
wait_for_serverapp_finish

SUCCESS=1
print_section "Run finished successfully."

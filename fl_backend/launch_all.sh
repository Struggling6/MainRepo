#!/bin/bash
set -euo pipefail

BASE_DIR="/ceph/home/student.aau.dk/rr68qj/fl_backend"
LOG_DIR="${BASE_DIR}/logs"
FLWR_HOME="/ceph/home/student.aau.dk/rr68qj/.flwr"

cd "$BASE_DIR"
mkdir -p "$LOG_DIR"

if [ $# -lt 1 ]; then
  echo "Usage: bash launch_all.sh <NUM_CLIENTS>"
  exit 1
fi

NUM_CLIENTS="$1"

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

  echo "Waiting a few seconds for ports/resources to release..."
  sleep 5
}

cleanup_logs() {
  print_section "Cleaning old log files..."
  rm -f "${LOG_DIR}"/superlink-*.out "${LOG_DIR}"/superlink-*.err \
        "${LOG_DIR}"/supernode-*.out "${LOG_DIR}"/supernode-*.err
}

check_datasets() {
  print_section "Checking dataset files..."

  for ((i=1; i<=NUM_CLIENTS; i++)); do
    DATASET_PATH="${BASE_DIR}/datasets/data${i}.csv"
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

wait_for_superlink_running() {
  echo "Waiting for SuperLink job to start running..."

  for _ in {1..60}; do
    JOB_STATE="$(squeue -j "$SUPERLINK_JOB_ID" -h -o "%T" || true)"

    if [ "$JOB_STATE" = "RUNNING" ]; then
      echo "SuperLink job is RUNNING."
      return 0
    fi

    if [ -z "$JOB_STATE" ]; then
      echo "SuperLink job disappeared before reaching RUNNING."
      sacct -j "$SUPERLINK_JOB_ID" --format=JobID,State,ExitCode || true
      exit 1
    fi

    sleep 2
  done

  echo "Timed out waiting for SuperLink job to start."
  sacct -j "$SUPERLINK_JOB_ID" --format=JobID,State,ExitCode || true
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
  echo "Check: $SUPERLINK_LOG"
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
      echo
      echo "--- superlink stdout ---"
      cat "$SUPERLINK_LOG" 2>/dev/null || true
      echo
      echo "--- superlink stderr ---"
      cat "$SUPERLINK_ERR_LOG" 2>/dev/null || true
      exit 1
    fi

    sleep 2
  done

  if [ "$READY" -ne 1 ]; then
    echo "SuperLink API did not become ready in time."
    echo
    echo "--- superlink stdout ---"
    cat "$SUPERLINK_LOG" 2>/dev/null || true
    echo
    echo "--- superlink stderr ---"
    cat "$SUPERLINK_ERR_LOG" 2>/dev/null || true
    exit 1
  fi

  echo "SuperLink is ready."
}

submit_supernodes() {
  print_section "Submitting ${NUM_CLIENTS} SuperNode jobs..."

  SUPERNODE_JOB_IDS=()

  for ((i=1; i<=NUM_CLIENTS; i++)); do
    FACILITY_ID="client-${i}"
    DATA_PATH="/workspace/fl_backend/datasets/data${i}.csv"
    PARTITION_ID=$((i - 1))
    CLIENTAPPIO_PORT=$((9093 + i))   # 9094, 9095, 9096, ...

    echo "Submitting SuperNode for ${FACILITY_ID} using ${DATA_PATH}"
    echo "  -> ClientAppIo port: ${CLIENTAPPIO_PORT}"
    echo "  -> Partition ID: ${PARTITION_ID}"

    JOB_ID=$(sbatch --parsable supernode.sbatch "$SUPERLINK_HOST" "$FACILITY_ID" "$DATA_PATH" "$PARTITION_ID" "$CLIENTAPPIO_PORT")
    echo "  -> SuperNode job ID: $JOB_ID"

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

    FINAL_STATE="$(squeue -j "$JOB_ID" -h -o "%T" || true)"
    if [ "$FINAL_STATE" != "RUNNING" ]; then
      echo "Timed out waiting for SuperNode job ${JOB_ID} to become RUNNING."
      sacct -j "$JOB_ID" --format=JobID,State,ExitCode,NodeList,Elapsed || true
      exit 1
    fi
  done

  echo "All SuperNode jobs are RUNNING."
}

run_flower() {
  print_section "Launching Flower run..."
  bash run_serverapp.sh "$SUPERLINK_HOST"
}

cleanup_old_jobs
cleanup_logs
check_datasets
update_num_clients_in_config

print_section "Submitting SuperLink job..."
SUPERLINK_JOB_ID=$(sbatch --parsable superlink.sbatch)
echo "SuperLink job ID: $SUPERLINK_JOB_ID"

SUPERLINK_LOG="${LOG_DIR}/superlink-${SUPERLINK_JOB_ID}.out"
SUPERLINK_ERR_LOG="${LOG_DIR}/superlink-${SUPERLINK_JOB_ID}.err"

wait_for_superlink_running
wait_for_superlink_host
wait_for_superlink_ready

submit_supernodes
wait_for_supernodes_running

echo "Waiting a few extra seconds before starting Flower run..."
sleep 10

run_flower
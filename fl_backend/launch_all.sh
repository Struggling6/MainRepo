#!/bin/bash
set -euo pipefail

BASE_DIR="/ceph/home/student.aau.dk/rr68qj/fl_backend"
LOG_DIR="${BASE_DIR}/logs"
CONFIG_PY="${BASE_DIR}/config.py"

cd "$BASE_DIR"
mkdir -p "$LOG_DIR"

if [ $# -lt 1 ]; then
  echo "Usage: bash launch_all.sh <NUM_CLIENTS>"
  exit 1
fi

NUM_CLIENTS="$1"

if ! [[ "$NUM_CLIENTS" =~ ^[0-9]+$ ]] || [ "$NUM_CLIENTS" -lt 1 ]; then
  echo "NUM_CLIENTS must be a positive integer"
  exit 1
fi

echo "========================================"
echo "Cleaning up old Flower jobs..."
echo "========================================"

for jobname in fl-superlink fl-supernode fl-serverapp; do
  JOB_IDS=$(squeue --me --noheader --format="%A %j" | awk -v name="$jobname" '$2 == name {print $1}')
  if [ -n "${JOB_IDS:-}" ]; then
    echo "Cancelling old jobs for ${jobname}:"
    echo "$JOB_IDS"
    scancel $JOB_IDS || true
  fi
done

echo "Waiting a few seconds for ports/resources to release..."
sleep 5

echo "========================================"
echo "Cleaning old log files..."
echo "========================================"

find "$LOG_DIR" -maxdepth 1 -type f \( -name "superlink-*.out" -o -name "superlink-*.err" -o -name "supernode-*.out" -o -name "supernode-*.err" \) -delete

echo "========================================"
echo "Checking dataset files..."
echo "========================================"

for ((i=1; i<=NUM_CLIENTS; i++)); do
  HOST_DATA_FILE="${BASE_DIR}/datasets/data${i}.csv"
  if [ ! -f "$HOST_DATA_FILE" ]; then
    echo "Missing dataset file: $HOST_DATA_FILE"
    exit 1
  fi
  echo "Found: $HOST_DATA_FILE"
done

echo "========================================"
echo "Updating config.py num_clients -> ${NUM_CLIENTS}"
echo "========================================"

python3 - <<PY
from pathlib import Path
import re

config_path = Path(r"$CONFIG_PY")
text = config_path.read_text(encoding="utf-8")

pattern = r'("num_clients"\s*:\s*)\d+'
new_text, count = re.subn(pattern, rf'\g<1>{int("$NUM_CLIENTS")}', text, count=1)

if count != 1:
    raise RuntimeError("Could not find exactly one num_clients entry in config.py")

config_path.write_text(new_text, encoding="utf-8")
print("Updated num_clients successfully.")
PY

echo "========================================"
echo "Submitting SuperLink job..."
echo "========================================"

SUPERLINK_JOB_ID=$(sbatch --parsable superlink.sbatch)
echo "SuperLink job ID: $SUPERLINK_JOB_ID"

SUPERLINK_LOG="${LOG_DIR}/superlink-${SUPERLINK_JOB_ID}.out"
SUPERLINK_ERR_LOG="${LOG_DIR}/superlink-${SUPERLINK_JOB_ID}.err"

echo "Waiting for SuperLink job to start running..."
while true; do
  STATE=$(squeue -j "$SUPERLINK_JOB_ID" --noheader --format="%T" || true)

  if [ "$STATE" = "RUNNING" ]; then
    echo "SuperLink job is RUNNING."
    break
  fi

  if ! squeue -j "$SUPERLINK_JOB_ID" --noheader >/dev/null 2>&1 || [ -z "$STATE" ]; then
    echo "SuperLink job disappeared before starting."
    sacct -j "$SUPERLINK_JOB_ID" --format=JobID,State,ExitCode
    exit 1
  fi

  sleep 2
done

echo "Waiting for SuperLink host to appear in log..."
for _ in {1..60}; do
  if [ -f "$SUPERLINK_LOG" ] && grep -q "Starting SuperLink on host:" "$SUPERLINK_LOG"; then
    break
  fi
  sleep 2
done

if [ ! -f "$SUPERLINK_LOG" ]; then
  echo "SuperLink log not found: $SUPERLINK_LOG"
  exit 1
fi

SUPERLINK_HOST=$(grep "Starting SuperLink on host:" "$SUPERLINK_LOG" | tail -n 1 | awk '{print $5}')

if [ -z "${SUPERLINK_HOST}" ]; then
  echo "Failed to determine SuperLink host from log."
  echo "Check: $SUPERLINK_LOG"
  exit 1
fi

echo "SuperLink host detected: $SUPERLINK_HOST"

echo "Waiting for SuperLink API to become ready..."
for _ in {1..90}; do
  if [ -f "$SUPERLINK_ERR_LOG" ] && grep -q "Starting Control API on 0.0.0.0:9093" "$SUPERLINK_ERR_LOG"; then
    echo "SuperLink is ready."
    break
  fi

  if ! squeue -j "$SUPERLINK_JOB_ID" --noheader >/dev/null 2>&1 && ! grep -q "Starting Control API on 0.0.0.0:9093" "$SUPERLINK_ERR_LOG" 2>/dev/null; then
    echo "SuperLink job disappeared before becoming ready."
    sacct -j "$SUPERLINK_JOB_ID" --format=JobID,State,ExitCode
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

if ! grep -q "Starting Control API on 0.0.0.0:9093" "$SUPERLINK_ERR_LOG" 2>/dev/null; then
  echo "SuperLink did not become ready in time."
  exit 1
fi

echo "========================================"
echo "Submitting ${NUM_CLIENTS} SuperNode jobs..."
echo "========================================"

for ((i=1; i<=NUM_CLIENTS; i++)); do
  FACILITY_ID="client-${i}"
  DATA_PATH="/workspace/fl_backend/datasets/data${i}.csv"
  PARTITION_ID=0
  CLIENTAPPIO_PORT=$((9093 + i))

  echo "Submitting SuperNode for ${FACILITY_ID} using ${DATA_PATH}"
  echo "  -> ClientAppIo port: ${CLIENTAPPIO_PORT}"
  JOB_ID=$(sbatch --parsable supernode.sbatch "$SUPERLINK_HOST" "$FACILITY_ID" "$DATA_PATH" "$PARTITION_ID" "$CLIENTAPPIO_PORT")
  echo "  -> SuperNode job ID: $JOB_ID"
done

echo "Waiting before starting Flower run..."
sleep 15

echo "========================================"
echo "Launching Flower run..."
echo "========================================"

bash run_serverapp.sh "$SUPERLINK_HOST"
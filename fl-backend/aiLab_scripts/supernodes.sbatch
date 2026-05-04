#!/bin/bash
#SBATCH --job-name=fl-supernodes
#SBATCH --mem=24G
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00

set -euo pipefail

echo "Running SuperNode bundle"
echo "Host: $(hostname)"
echo "Clients: ${CLIENT_START}-${CLIENT_END}"

PIDS=()

cleanup() {
  echo "Cleaning up..."
  for pid in "${PIDS[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
  wait || true
}

trap cleanup EXIT INT TERM

for ((i=CLIENT_START; i<=CLIENT_END; i++)); do
  PORT=$((9093 + i))
  PARTITION=$((i - 1))
  GPU=$(((i - CLIENT_START) % 4))

  echo "Starting client $i on GPU $GPU"

  singularity exec --nv \
    -B "${VENV_DIR}:${VENV_DIR}" \
    -B "${BASE_DIR}:/workspace/fl-backend" \
    "${CONTAINER}" \
    /bin/bash -lc "
      export CUDA_VISIBLE_DEVICES=${GPU}
      source ${VENV_DIR}/bin/activate
      cd /workspace/fl-backend

      exec ${VENV_DIR}/bin/flower-supernode \
        --insecure \
        --superlink ${SUPERLINK_HOST}:9092 \
        --clientappio-api-address 127.0.0.1:${PORT} \
        --node-config 'partition-id=${PARTITION} num-partitions=${NUM_CLIENTS}'
    " > "${LOG_DIR}/supernode-${SLURM_JOB_ID}-${i}.out" \
      2> "${LOG_DIR}/supernode-${SLURM_JOB_ID}-${i}.err" &

  PIDS+=("$!")
  sleep 2
done

wait
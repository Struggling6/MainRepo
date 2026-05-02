#!/bin/bash
#SBATCH --job-name=fl-supernode
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00

set -euo pipefail

mkdir -p "${LOG_DIR}"

if [ $# -lt 5 ]; then
  echo "Usage: sbatch supernode.sbatch <SUPERLINK_HOST> <FACILITY_ID> <DATA_PATH> <PARTITION_ID> <CLIENTAPPIO_PORT>"
  exit 1
fi

if [ -z "${NUM_CLIENTS:-}" ]; then
  echo "NUM_CLIENTS is not set."
  exit 1
fi

SUPERLINK_HOST="$1"
FACILITY_ID="$2"
DATA_PATH="$3"
PARTITION_ID="$4"
CLIENTAPPIO_PORT="$5"

echo "Starting SuperNode on host: $(hostname)"
echo "Connecting to SuperLink at: ${SUPERLINK_HOST}:9092"
echo "Facility ID: ${FACILITY_ID}"
echo "Data path: ${DATA_PATH}"
echo "Partition ID: ${PARTITION_ID}"
echo "ClientAppIo port: ${CLIENTAPPIO_PORT}"
echo "Job ID: ${SLURM_JOB_ID}"
echo "BASE_DIR=${BASE_DIR}"
echo "VENV_DIR=${VENV_DIR}"
echo "NUM_CLIENTS=${NUM_CLIENTS}"

srun singularity exec --nv \
  -B "${VENV_DIR}:${VENV_DIR}" \
  -B "${BASE_DIR}:/workspace/fl-backend" \
  /ceph/container/pytorch/pytorch_25.08.sif \
  /bin/bash -lc "
    source \"${VENV_DIR}/bin/activate\"
    export PYTHONWARNINGS='ignore:Protobuf gencode version:UserWarning'
    export PYTHONUNBUFFERED=1
    cd /workspace/fl-backend
    exec \"${VENV_DIR}/bin/flower-supernode\" \
      --insecure \
      --superlink \"${SUPERLINK_HOST}:9092\" \
      --clientappio-api-address \"127.0.0.1:${CLIENTAPPIO_PORT}\" \
      --node-config 'partition-id=${PARTITION_ID} num-partitions=${NUM_CLIENTS} facility-id=\"${FACILITY_ID}\" data-path=\"${DATA_PATH}\"'
  "
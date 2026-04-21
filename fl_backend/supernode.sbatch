+#!/bin/bash
#SBATCH --job-name=fl-supernode
#SBATCH --output=/ceph/home/student.aau.dk/rr68qj/fl_backend/logs/supernode-%j.out
#SBATCH --error=/ceph/home/student.aau.dk/rr68qj/fl_backend/logs/supernode-%j.err
#SBATCH --chdir=/ceph/home/student.aau.dk/rr68qj/fl_backend
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00

set -euo pipefail

mkdir -p /ceph/home/student.aau.dk/rr68qj/fl_backend/logs

if [ $# -lt 5 ]; then
  echo "Usage: sbatch supernode.sbatch <SUPERLINK_HOST> <FACILITY_ID> <DATA_PATH> <PARTITION_ID> <CLIENTAPPIO_PORT>"
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

srun singularity exec \
  -B /ceph/home/student.aau.dk/rr68qj/fl_venv:/scratch/fl_venv \
  -B /ceph/home/student.aau.dk/rr68qj/fl_backend:/workspace/fl_backend \
  /ceph/container/pytorch/pytorch_25.08.sif \
  /bin/bash -lc "
    source /scratch/fl_venv/bin/activate
    export PYTHONWARNINGS='ignore:Protobuf gencode version:UserWarning'
    cd /workspace/fl_backend
    /scratch/fl_venv/bin/flower-supernode \
      --insecure \
      --superlink ${SUPERLINK_HOST}:9092 \
      --clientappio-api-address 127.0.0.1:${CLIENTAPPIO_PORT} \
      --node-config 'partition-id=${PARTITION_ID} num-partitions=1 facility-id=\"${FACILITY_ID}\" data-path=\"${DATA_PATH}\"'
  "

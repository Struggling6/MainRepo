#!/bin/bash
#SBATCH --job-name=fl-superlink
#SBATCH --mem=4G
#SBATCH --cpus-per-task=2
#SBATCH --time=12:00:00

set -euo pipefail

mkdir -p "${LOG_DIR}"

echo "Starting SuperLink on host: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "BASE_DIR=${BASE_DIR}"
echo "VENV_DIR=${VENV_DIR}"

srun singularity exec \
  -B "${VENV_DIR}:${VENV_DIR}" \
  /ceph/container/pytorch/pytorch_25.08.sif \
  /bin/bash -lc "
    source \"${VENV_DIR}/bin/activate\"
    export PYTHONWARNINGS='ignore:Protobuf gencode version:UserWarning'
    exec \"${VENV_DIR}/bin/flower-superlink\" --insecure
  "
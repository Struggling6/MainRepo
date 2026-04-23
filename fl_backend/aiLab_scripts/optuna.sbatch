#!/bin/bash
#SBATCH --job-name=optuna_search
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=15
#SBATCH --mem=24G
#SBATCH --time=04:00:00

set -euo pipefail

SCRIPT_DIR="${SCRIPT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
BASE_DIR="${BASE_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
PARENT_DIR="$(cd "${BASE_DIR}/.." && pwd)"

VENV_DIR="${VENV_DIR:-${PARENT_DIR}/fl_venv}"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/logs}"
CONTAINER="${CONTAINER:-/ceph/container/pytorch/pytorch_25.08.sif}"

mkdir -p "${LOG_DIR}"

echo "Optuna trial starting"
echo "  Array task : ${SLURM_ARRAY_TASK_ID:-N/A}"
echo "  Job ID     : ${SLURM_JOB_ID}"
echo "  Node       : $(hostname)"
echo "  SCRIPT_DIR : ${SCRIPT_DIR}"
echo "  BASE_DIR   : ${BASE_DIR}"
echo "  VENV_DIR   : ${VENV_DIR}"
echo "  LOG_DIR    : ${LOG_DIR}"

srun singularity exec --nv \
  -B "${VENV_DIR}:${VENV_DIR}" \
  -B "${BASE_DIR}:/workspace/fl_backend" \
  "${CONTAINER}" \
  /bin/bash -lc "
    source \"${VENV_DIR}/bin/activate\"
    export PYTHONUNBUFFERED=1
    export PYTHONWARNINGS='ignore:Protobuf gencode version:UserWarning'
    cd /workspace/fl_backend
    exec python -m training.optimize \
      --trials 1 \
      --epochs \"${OPTUNA_EPOCHS}\" \
      --patience \"${OPTUNA_PATIENCE}\" \
      --storage \"${OPTUNA_STORAGE}\" \
      --study-name \"${OPTUNA_STUDY}\"
  "

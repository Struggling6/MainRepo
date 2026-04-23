#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PARENT_DIR="$(cd "${BASE_DIR}/.." && pwd)"

VENV_DIR="${VENV_DIR:-${PARENT_DIR}/fl_venv}"
CONTAINER="${CONTAINER:-/ceph/container/pytorch/pytorch_25.08.sif}"
STORAGE="${STORAGE:-sqlite:////${BASE_DIR#/}/optuna_study.db}"
STUDY_NAME="${STUDY_NAME:-lead_anomaly_detection}"

echo "Getting best Optuna parameters"
echo "  BASE_DIR   : ${BASE_DIR}"
echo "  VENV_DIR   : ${VENV_DIR}"
echo "  CONTAINER  : ${CONTAINER}"
echo "  STORAGE    : ${STORAGE}"
echo "  STUDY_NAME : ${STUDY_NAME}"

srun singularity exec --nv \
  -B "${VENV_DIR}:${VENV_DIR}" \
  -B "${BASE_DIR}:/workspace/fl_backend" \
  "${CONTAINER}" \
  /bin/bash -lc "
    source \"${VENV_DIR}/bin/activate\"
    export PYTHONUNBUFFERED=1
    export PYTHONWARNINGS='ignore:Protobuf gencode version:UserWarning'
    cd /workspace/fl_backend
    exec python training/get_best_params.py \
      --storage \"${STORAGE}\" \
      --study-name \"${STUDY_NAME}\"
  "

#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PARENT_DIR="$(cd "${BASE_DIR}/.." && pwd)"

VENV_DIR="${VENV_DIR:-${PARENT_DIR}/fl_venv}"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/logs}"
CONTAINER="${CONTAINER:-/ceph/container/pytorch/pytorch_25.08.sif}"
STORAGE="${STORAGE:-sqlite:////${BASE_DIR#/}/optuna_study.db}"
STUDY_NAME="${STUDY_NAME:-lead_anomaly_detection}"
MAX_PARALLEL="${MAX_PARALLEL:-8}"

# ── Defaults ──────────────────────────────────────────────────────────── #
TRIALS=50
EPOCHS=10
PATIENCE=10

# ── Argument parsing ──────────────────────────────────────────────────── #
while [[ $# -gt 0 ]]; do
    case "$1" in
        --trials|-t)   TRIALS="$2"; shift 2 ;;
        --epochs|-e)   EPOCHS="$2"; shift 2 ;;
        --patience|-p) PATIENCE="$2"; shift 2 ;;
        --study-name|-s) STUDY_NAME="$2"; shift 2 ;;
        --max-parallel|-m) MAX_PARALLEL="$2"; shift 2 ;;
        --storage) STORAGE="$2"; shift 2 ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: bash run_optuna.sh [--trials N] [--epochs N] [--patience N] [--study-name NAME] [--max-parallel N] [--storage URI]"
            exit 1
            ;;
    esac
done

print_section() {
    echo "========================================"
    echo "$1"
    echo "========================================"
}

mkdir -p "${LOG_DIR}"

print_section "Optuna Hyperparameter Search"
echo "  Trials       : ${TRIALS}"
echo "  Epochs       : ${EPOCHS}"
echo "  Patience     : ${PATIENCE}"
echo "  Study name   : ${STUDY_NAME}"
echo "  Storage      : ${STORAGE}"
echo "  Max parallel : ${MAX_PARALLEL}"
echo "  BASE_DIR     : ${BASE_DIR}"
echo "  VENV_DIR     : ${VENV_DIR}"
echo "  LOG_DIR      : ${LOG_DIR}"

cd "${SCRIPT_DIR}"

# ── Submit array jobs — each runs optimize.py --trials 1 ──────────────── #
REMAINING=$TRIALS
BATCH=1

print_section "Submitting ${TRIALS} trials in batches of ${MAX_PARALLEL}..."

while [ "$REMAINING" -gt 0 ]; do
    if [ "$REMAINING" -ge "$MAX_PARALLEL" ]; then
        BATCH_SIZE=$MAX_PARALLEL
    else
        BATCH_SIZE=$REMAINING
    fi

    ARRAY_END=$((BATCH_SIZE - 1))

    echo ""
    echo "Submitting batch ${BATCH}: ${BATCH_SIZE} trials (array 0-${ARRAY_END})"

    JOB_ID=$(sbatch --parsable \
        --array=0-"${ARRAY_END}" \
        --chdir="${BASE_DIR}" \
        --output="${LOG_DIR}/optuna_%A_%a.out" \
        --error="${LOG_DIR}/optuna_%A_%a.err" \
        --export=ALL,BASE_DIR="${BASE_DIR}",VENV_DIR="${VENV_DIR}",LOG_DIR="${LOG_DIR}",CONTAINER="${CONTAINER}",OPTUNA_EPOCHS="${EPOCHS}",OPTUNA_PATIENCE="${PATIENCE}",OPTUNA_STORAGE="${STORAGE}",OPTUNA_STUDY="${STUDY_NAME}" \
        "${SCRIPT_DIR}/optuna.sbatch")

    echo "  -> Job ID: ${JOB_ID}"

    REMAINING=$((REMAINING - BATCH_SIZE))
    BATCH=$((BATCH + 1))
done

print_section "All ${TRIALS} trials submitted!"
echo ""
echo "Monitor:     squeue --me"
echo "Logs:        tail -f ${LOG_DIR}/optuna_<jobid>_0.out"
echo "Results:     bash ${SCRIPT_DIR}/get_best_params.sh"
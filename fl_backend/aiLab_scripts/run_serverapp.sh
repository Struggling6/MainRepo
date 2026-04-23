#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PARENT_DIR="$(cd "${BASE_DIR}/.." && pwd)"
VENV_DIR="${PARENT_DIR}/fl_venv"
LOG_DIR="${SCRIPT_DIR}/logs"
FLWR_HOME="${HOME}/.flwr"

if [ $# -lt 1 ]; then
  echo "Usage: bash ${SCRIPT_DIR}/run_serverapp.sh <SUPERLINK_HOST>"
  exit 1
fi

SUPERLINK_HOST="$1"
CONFIG_FILE="${FLWR_HOME}/config.toml"

mkdir -p "${FLWR_HOME}"
mkdir -p "${LOG_DIR}"

cat > "$CONFIG_FILE" <<EOF
[superlink]
default = "ailab"

[superlink.ailab]
address = "${SUPERLINK_HOST}:9093"
insecure = true
EOF

echo "Wrote Flower config to $CONFIG_FILE"
echo "Using SuperLink host: $SUPERLINK_HOST"
echo "BASE_DIR=${BASE_DIR}"
echo "VENV_DIR=${VENV_DIR}"

srun -N1 -n1 --ntasks=1 --mem=8G --cpus-per-task=1 --time=03:00:00 \
  singularity exec \
  -B "${VENV_DIR}:${VENV_DIR}" \
  -B "${BASE_DIR}:/workspace/fl_backend" \
  /ceph/container/pytorch/pytorch_25.08.sif \
  /bin/bash -lc "
    export FLWR_HOME=${FLWR_HOME}
    source ${VENV_DIR}/bin/activate
    export PYTHONWARNINGS='ignore:Protobuf gencode version:UserWarning'
    cd /workspace/fl_backend
    stdbuf -oL -eL ${VENV_DIR}/bin/flwr run . ailab --stream 2>&1 | tee /workspace/fl_backend/aiLab_scripts/logs/serverapp.log
  "

#!/bin/bash
#SBATCH --job-name=fl-serverapp
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00

set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: sbatch serverapp.sbatch <SUPERLINK_HOST>"
  exit 1
fi

SUPERLINK_HOST="$1"

mkdir -p "${LOG_DIR}"
mkdir -p "${FLWR_HOME}"

CONFIG_FILE="${FLWR_HOME}/config.toml"

cat > "${CONFIG_FILE}" <<EOF
[superlink]
default = "ailab"

[superlink.ailab]
address = "${SUPERLINK_HOST}:9093"
insecure = true
EOF

echo "Starting ServerApp on host: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "SUPERLINK_HOST=${SUPERLINK_HOST}"
echo "BASE_DIR=${BASE_DIR}"
echo "VENV_DIR=${VENV_DIR}"
echo "FLWR_HOME=${FLWR_HOME}"

srun singularity exec \
  -B "${VENV_DIR}:${VENV_DIR}" \
  -B "${BASE_DIR}:/workspace/fl-backend" \
  -B "${FLWR_HOME}:${FLWR_HOME}" \
  /ceph/container/pytorch/pytorch_25.08.sif \
  /bin/bash -lc "
    export FLWR_HOME=\"${FLWR_HOME}\"
    source \"${VENV_DIR}/bin/activate\"
    export PYTHONWARNINGS='ignore:Protobuf gencode version:UserWarning'
    export PYTHONUNBUFFERED=1
    cd /workspace/fl-backend
    exec stdbuf -oL -eL \"${VENV_DIR}/bin/flwr\" run . ailab --stream
  "
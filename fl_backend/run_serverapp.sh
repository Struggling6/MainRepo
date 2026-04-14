#!/bin/bash
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: bash run_serverapp.sh <SUPERLINK_HOST>"
  exit 1
fi

SUPERLINK_HOST="$1"
CONFIG_FILE="/ceph/home/student.aau.dk/rr68qj/.flwr/config.toml"

mkdir -p /ceph/home/student.aau.dk/rr68qj/.flwr
mkdir -p /ceph/home/student.aau.dk/rr68qj/fl_backend/logs

cat > "$CONFIG_FILE" <<EOF
[superlink]
default = "ailab"

[superlink.ailab]
address = "${SUPERLINK_HOST}:9093"
insecure = true
EOF

echo "Wrote Flower config to $CONFIG_FILE"
echo "Using SuperLink host: $SUPERLINK_HOST"

srun -N1 -n1 --ntasks=1 --mem=8G --cpus-per-task=1 --time=00:30:00 \
  singularity exec \
  -B /ceph/home/student.aau.dk/rr68qj/fl_venv:/scratch/fl_venv \
  -B /ceph/home/student.aau.dk/rr68qj/fl_backend:/workspace/fl_backend \
  /ceph/container/pytorch/pytorch_25.08.sif \
  /bin/bash -lc "
    export FLWR_HOME=/ceph/home/student.aau.dk/rr68qj/.flwr
    source /scratch/fl_venv/bin/activate
    export PYTHONWARNINGS='ignore:Protobuf gencode version:UserWarning'
    cd /workspace/fl_backend
    /scratch/fl_venv/bin/flwr run . ailab --stream
  "

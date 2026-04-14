#!/bin/bash
#SBATCH --job-name=fl-superlink
#SBATCH --output=/ceph/home/student.aau.dk/rr68qj/fl_backend/logs/superlink-%j.out
#SBATCH --error=/ceph/home/student.aau.dk/rr68qj/fl_backend/logs/superlink-%j.err
#SBATCH --chdir=/ceph/home/student.aau.dk/rr68qj/fl_backend
#SBATCH --mem=4G
#SBATCH --cpus-per-task=2
#SBATCH --time=01:00:00

set -euo pipefail

mkdir -p /ceph/home/student.aau.dk/rr68qj/fl_backend/logs

echo "Starting SuperLink on host: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID}"

srun singularity exec \
  -B /ceph/home/student.aau.dk/rr68qj/fl_venv:/scratch/fl_venv \
  /ceph/container/pytorch/pytorch_25.08.sif \
  /bin/bash -lc "
    source /scratch/fl_venv/bin/activate
    export PYTHONWARNINGS='ignore:Protobuf gencode version:UserWarning'
    /scratch/fl_venv/bin/flower-superlink --insecure
  "

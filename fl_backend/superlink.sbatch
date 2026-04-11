#!/bin/bash
#SBATCH --job-name=fl-superlink
#SBATCH --output=logs/superlink-%j.out
#SBATCH --error=logs/superlink-%j.err
#SBATCH --mem=4G
#SBATCH --cpus-per-task=2
#SBATCH --time=01:00:00

set -euo pipefail

mkdir -p logs

echo "Starting SuperLink on host: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID}"

srun -N1 -n1 --ntasks=1 singularity exec \
  -B "$HOME/fl_venv:/scratch/fl_venv" \
  /ceph/container/pytorch/pytorch_25.08.sif \
  /bin/bash -lc '
    source /scratch/fl_venv/bin/activate
    export PYTHONWARNINGS="ignore:Protobuf gencode version:UserWarning"
    flower-superlink --insecure
  '
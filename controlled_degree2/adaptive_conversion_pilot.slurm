#!/usr/bin/env bash
#SBATCH --account=MST114196
#SBATCH --job-name=quadratic-adaptive-pilot
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:H200:1
#SBATCH --mem=100G
#SBATCH --time=00:45:00
#SBATCH --output=adaptive-%j.out
#SBATCH --error=adaptive-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MXNET_CPU_WORKER_NTHREADS=1
export PYTHONUNBUFFERED=1 PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
: "${ADAPTIVE_OUTPUT:?}"
exec /home/u8798807/.conda/envs/face_recog/bin/python -m controlled_degree2.adaptive_conversion_pilot \
  --output "$ADAPTIVE_OUTPUT" "$@"

#!/usr/bin/env bash
#SBATCH --account=MST114196
#SBATCH --job-name=shared-d2-holdout
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H200:1
#SBATCH --mem=100G
#SBATCH --time=2:00:00
#SBATCH --output=shared-holdout-%j.out
#SBATCH --error=shared-holdout-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
: "${CHECKPOINT:?}"
: "${VALIDATION_OUTPUT:?}"
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MXNET_CPU_WORKER_NTHREADS=1 PYTHONUNBUFFERED=1
/home/u8798807/.conda/envs/face_recog/bin/python -m controlled_degree2.validate_shared \
  --checkpoint "$CHECKPOINT" --output "$VALIDATION_OUTPUT"

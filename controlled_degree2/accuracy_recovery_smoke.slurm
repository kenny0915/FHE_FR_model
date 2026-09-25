#!/usr/bin/env bash
#SBATCH --account=MST114196
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:H200:4
#SBATCH --mem=160G
#SBATCH --time=00:30:00
set -euo pipefail
cd "${ACCURACY_CODE_ROOT:-${SLURM_SUBMIT_DIR:?}}"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MXNET_CPU_WORKER_NTHREADS=1 PYTHONUNBUFFERED=1
python -m controlled_degree2.accuracy_recovery_smoke \
  --output-root "${SMOKE_ROOT:?}" --checkpoint "${CHECKPOINT:?}" \
  --teacher "${TEACHER:?}" --dataset-root "${DATASET_ROOT:?}" \
  --canary-root "${CANARY_ROOT:?}"

#!/usr/bin/env bash
#SBATCH --account=MST114196
#SBATCH --job-name=control-bts-layouts
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
set -euo pipefail
cd "${BTS_CODE_ROOT:-${SLURM_SUBMIT_DIR:?}}"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MXNET_CPU_WORKER_NTHREADS=1 PYTHONUNBUFFERED=1
PYTHON_BIN="${ACCURACY_PYTHON:-${HOME}/.conda/envs/face_recog/bin/python}"
"$PYTHON_BIN" -m controlled_degree2.control_bts_layouts \
  --checkpoint "${CHECKPOINT:?}" --dataset-root "${DATASET_ROOT:?}" \
  --output-root "${BTS_OUTPUT_ROOT:?}"

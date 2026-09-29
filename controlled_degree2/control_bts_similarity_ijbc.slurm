#!/usr/bin/env bash
#SBATCH --account=MST114196
#SBATCH --job-name=bts6-similarity
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=04:00:00
set -euo pipefail
cd "${BTS_CODE_ROOT:-${SLURM_SUBMIT_DIR:?}}"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
PYTHON_BIN="${ACCURACY_PYTHON:-${HOME}/.conda/envs/face_recog/bin/python}"
: "${CHECKPOINT:?}"
: "${IJBC_ROOT:?}"
: "${SIMILARITY_OUTPUT:?}"
mkdir -p "$SIMILARITY_OUTPUT"
"$PYTHON_BIN" eval_ijbc.py \
  --model-prefix "$CHECKPOINT" --image-path "$IJBC_ROOT" \
  --result-dir "$SIMILARITY_OUTPUT" --job bts6_zero_failed \
  --network r50_controlled_d2 --target IJBC --batch-size 256 \
  --image-workers 8 --similarity both \
  --bts-failure-report "$SIMILARITY_OUTPUT/failures"

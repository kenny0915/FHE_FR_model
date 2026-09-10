#!/usr/bin/env bash
#SBATCH --account=MST114196
#SBATCH --job-name=ijbc-controlled-d2
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH --time=4:00:00
set -euo pipefail
cd /work/u8798807/FHE_FR_model
: "${CHECKPOINT:?Set CHECKPOINT}"; : "${RESULT_DIR:?Set RESULT_DIR}"
test -f "${CHECKPOINT}"
mkdir -p "${RESULT_DIR}"
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=12 PYTHONUNBUFFERED=1
/home/u8798807/.conda/envs/face_recog/bin/python eval_ijbc.py \
  --model-prefix "${CHECKPOINT}" --image-path ijb/IJBC \
  --result-dir "${RESULT_DIR}" --batch-size 512 \
  --job "${EVAL_JOB_NAME:-controlled_d2}" --target IJBC \
  --network "${NETWORK:-r50_controlled_d2}" \
  --nonfinite-manifest "${RESULT_DIR}/nonfinite_manifest.csv"

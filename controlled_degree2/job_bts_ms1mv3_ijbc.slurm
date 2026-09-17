#!/usr/bin/env bash
#SBATCH --account=MST114196
#SBATCH --job-name=bts-ms1k-ijbc
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=04:00:00
set -euo pipefail
cd /work/u8798807/FHE_FR_model
: "${RUN_DIR:?Set a fresh RUN_DIR}"
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
PYTHON=/home/u8798807/.conda/envs/face_recog/bin/python
if [[ -z "${SCALED_CHECKPOINT:-}" ]]; then
"${PYTHON}" -m controlled_degree2.calibrate_bts_ms1mv3 \
  --checkpoint work_dirs/controlled_degree2_tail_ms1mv3_20260907/progressive/student_best.pt \
  --output-dir "${RUN_DIR}/scaled" --dataset-root ms1m-retinaface-t1 \
  --images 1000 --seed 42 --target-absmax 0.8 --batch-size 64
SCALED_CHECKPOINT="${RUN_DIR}/scaled/student_scaled.pt"
fi
EVALUATION_DIR="${RUN_DIR}/${EVAL_SUBDIR:-ijbc}"
"${PYTHON}" eval_ijbc.py \
  --model-prefix "${SCALED_CHECKPOINT}" \
  --image-path ijb/IJBC --result-dir "${EVALUATION_DIR}" \
  --batch-size 256 --image-workers 8 --job ms1k_bts_zero_failed --target IJBC \
  --network r50_controlled_d2 --bts-failure-report "${EVALUATION_DIR}/bts"

#!/usr/bin/env bash
#SBATCH --account=MST114196
#SBATCH --job-name=progressive-unscaled-ijbc
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=04:00:00
set -euo pipefail
cd /work/u8798807/FHE_FR_model
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
/home/u8798807/.conda/envs/face_recog/bin/python eval_ijbc.py \
 --model-prefix work_dirs/controlled_degree2_tail_ms1mv3_20260907/progressive/student_best.pt \
 --image-path ijb/IJBC --result-dir work_dirs/progressive_best_unscaled_ijbc_20260919/result \
 --batch-size 256 --image-workers 8 --job unscaled_nonfinite_only --target IJBC \
 --network r50_controlled_d2 \
 --bts-failure-report work_dirs/progressive_best_unscaled_ijbc_20260919/result/failures \
 --ignore-bts-range

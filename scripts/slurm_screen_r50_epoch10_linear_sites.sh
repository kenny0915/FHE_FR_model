#!/bin/bash
#SBATCH --account=MST114196
#SBATCH --job-name=r50-e10-linear-sites
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --time=02:00:00
#SBATCH --output=work_dirs/ms1mv3_r50_herpn_epoch10_linear_sites/slurm-%j.out
#SBATCH --error=work_dirs/ms1mv3_r50_herpn_epoch10_linear_sites/slurm-%j.err

set -euo pipefail

cd /work/u8798807/FHE_FR_model
output=work_dirs/ms1mv3_r50_herpn_epoch10_linear_sites
mkdir -p "${output}"
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=12
export PYTHONUNBUFFERED=1

/home/u8798807/.conda/envs/face_recog/bin/python \
    eval/screen_linear_replacement_sites.py \
    --checkpoint work_dirs/ms1mv3_r50_herpn/model_epoch_10.pt \
    --batch-size 200 \
    --output "${output}/screen.json" \
    --checkpoint-output "${output}/model_linear9_selected.pt"

#!/bin/bash
#SBATCH --account=MST114196
#SBATCH --job-name=r50-e10-chlinear9
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --time=02:00:00
#SBATCH --output=work_dirs/ms1mv3_r50_herpn_epoch10_channelwise_linear9/slurm-%j.out
#SBATCH --error=work_dirs/ms1mv3_r50_herpn_epoch10_channelwise_linear9/slurm-%j.err

set -euo pipefail

cd /work/u8798807/FHE_FR_model
output=work_dirs/ms1mv3_r50_herpn_epoch10_channelwise_linear9
mkdir -p "${output}"
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=16
export PYTHONUNBUFFERED=1

/home/u8798807/.conda/envs/face_recog/bin/python \
    eval/screen_channelwise_linear_replacement.py \
    --checkpoint \
    work_dirs/ms1mv3_r50_herpn_epoch10_linear9_layer42/model_epoch_01.pt \
    --batch-size 200 \
    --output "${output}/screen.json" \
    --checkpoint-output "${output}/model_linear9_channelwise.pt"

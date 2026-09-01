#!/bin/bash
#SBATCH --account=MST114196
#SBATCH --job-name=r50-herpn-full-p1
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --exclude=25a-hgpn[143-145]
#SBATCH --mem=300G
#SBATCH --time=2-00:00:00
#SBATCH --output=work_dirs/ms1mv3_r50_herpn_full_conversion_phase1/slurm-%j.out
#SBATCH --error=work_dirs/ms1mv3_r50_herpn_full_conversion_phase1/slurm-%j.err

set -euo pipefail

cd /work/u8798807/FHE_FR_model
output=work_dirs/ms1mv3_r50_herpn_full_conversion_phase1
mkdir -p "${output}"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1

/home/u8798807/.conda/envs/face_recog/bin/torchrun \
    --standalone \
    --nproc_per_node=4 \
    train_v2.py \
    configs/ms1mv3_r50_no_relu_full_conversion_phase1

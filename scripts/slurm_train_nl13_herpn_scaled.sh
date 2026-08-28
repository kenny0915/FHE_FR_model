#!/bin/bash
#SBATCH --account=MST114196
#SBATCH --job-name=r50-nl13-scaled
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:8
#SBATCH --time=2-00:00:00
#SBATCH --output=work_dirs/ms1mv3_r50_nl13_prelu_herpn_scaled/slurm-%j.out
#SBATCH --error=work_dirs/ms1mv3_r50_nl13_prelu_herpn_scaled/slurm-%j.err

set -euo pipefail

cd /work/u8798807/FHE_FR_model
mkdir -p work_dirs/ms1mv3_r50_nl13_prelu_herpn_scaled
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=4

/home/u8798807/.conda/envs/face_recog/bin/torchrun \
    --standalone \
    --nproc_per_node=8 \
    train_v2.py configs/ms1mv3_r50_nl13_prelu_herpn_scaled

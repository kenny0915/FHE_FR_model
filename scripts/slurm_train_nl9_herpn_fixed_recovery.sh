#!/bin/bash
#SBATCH --account=MST114196
#SBATCH --job-name=r50-nl9-poly-recover
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:8
#SBATCH --time=1-00:00:00
#SBATCH --output=work_dirs/nl9-fixed-recovery-%j.out
#SBATCH --error=work_dirs/nl9-fixed-recovery-%j.err

set -euo pipefail

cd /work/u8798807/FHE_FR_model

config="${1:-configs/ms1mv3_r50_nl9_prelu_herpn_fixed_recovery}"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=4

/home/u8798807/.conda/envs/face_recog/bin/torchrun \
  --standalone \
  --nproc_per_node=8 \
  train_v2.py "${config}"

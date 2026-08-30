#!/bin/bash
#SBATCH --account=MST114196
#SBATCH --job-name=pillar-espn-smoke
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --exclude=25a-hgpn[143-144]
#SBATCH --cpus-per-task=64
#SBATCH --mem=500G
#SBATCH --time=02:00:00
#SBATCH --output=work_dirs/casia_r50_pillar_espn_smoke/slurm-%j.out
#SBATCH --error=work_dirs/casia_r50_pillar_espn_smoke/slurm-%j.err

set -euo pipefail

cd /work/u8798807/FHE_FR_model
mkdir -p work_dirs/casia_r50_pillar_espn_smoke
ml load miniconda3
conda activate face_recog

export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

/home/u8798807/.conda/envs/face_recog/bin/torchrun \
  --standalone \
  --nproc_per_node=8 \
  train_v2.py configs/casia_r50_pillar_espn_smoke

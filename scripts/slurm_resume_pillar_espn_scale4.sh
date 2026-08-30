#!/bin/bash
#SBATCH --account=MST114196
#SBATCH --job-name=pillar-scale4
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --exclude=25a-hgpn[143-144]
#SBATCH --cpus-per-task=64
#SBATCH --mem=500G
#SBATCH --time=2-00:00:00
#SBATCH --output=work_dirs/ms1mv3_r50_pillar_espn_d4_scale4/resume-%j.out
#SBATCH --error=work_dirs/ms1mv3_r50_pillar_espn_d4_scale4/resume-%j.err

set -euo pipefail

cd /work/u8798807/FHE_FR_model
ml load miniconda3
conda activate face_recog

export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

/home/u8798807/.conda/envs/face_recog/bin/torchrun \
  --standalone \
  --nproc_per_node=8 \
  train_v2.py configs/ms1mv3_r50_pillar_espn_scale4

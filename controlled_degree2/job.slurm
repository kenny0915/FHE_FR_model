#!/usr/bin/env bash
#SBATCH --account=MST114196
#SBATCH --job-name=controlled-d2
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --output=controlled-d2-%j.out
#SBATCH --error=controlled-d2-%j.err

set -euo pipefail
ml load miniconda3
conda activate face_recog

# Export these paths with sbatch --export=ALL,TEACHER_CKPT=...,RUN10_CKPT=...,
# DATASET_ROOT=...,OUTPUT_ROOT=... .  The reference checkpoint is never written.
bash controlled_degree2/run_controlled_degree2.sh "${STAGE:-all}"

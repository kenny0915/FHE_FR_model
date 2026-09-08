#!/usr/bin/env bash
#SBATCH --account=MST114196
#SBATCH --job-name=controlled-d2-refine
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
set -euo pipefail
ml load miniconda3
conda activate face_recog
bash controlled_degree2/run_tail_refine.sh

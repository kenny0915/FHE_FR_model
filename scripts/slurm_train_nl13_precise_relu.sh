#!/bin/bash
#SBATCH --account=MST114196
#SBATCH --job-name=nl13-precise-relu
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH --mem=500G
#SBATCH --time=2-00:00:00
#SBATCH --output=work_dirs/slurm-%x-%j.out
#SBATCH --error=work_dirs/slurm-%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=910915kenny@gmail.com

set -euo pipefail

cd /work/u8798807/FHE_FR_model

if [[ $# -ne 1 ]]; then
    echo "usage: sbatch $0 configs/<nl13-precise-relu-config>" >&2
    exit 2
fi

config="$1"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1

/home/u8798807/.conda/envs/face_recog/bin/torchrun \
    --standalone \
    --nproc_per_node=8 \
    train_v2.py "${config}"

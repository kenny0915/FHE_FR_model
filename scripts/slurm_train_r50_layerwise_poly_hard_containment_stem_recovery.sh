#!/bin/bash
#SBATCH --account=MST114196
#SBATCH --job-name=r50-poly-stem-recover
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH --exclude=25a-hgpn[143-145]
#SBATCH --mem=500G
#SBATCH --time=06:00:00
#SBATCH --output=work_dirs/ms1mv3_r50_layerwise_poly_hard_containment_stem/slurm-%j.out
#SBATCH --error=work_dirs/ms1mv3_r50_layerwise_poly_hard_containment_stem/slurm-%j.err

set -euo pipefail

cd /work/u8798807/FHE_FR_model
output=work_dirs/ms1mv3_r50_layerwise_poly_hard_containment_stem
mkdir -p "${output}"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1

/home/u8798807/.conda/envs/face_recog/bin/torchrun \
    --standalone \
    --nproc_per_node=8 \
    train_v2.py \
    configs/ms1mv3_r50_layerwise_poly_hard_containment_stem_recovery

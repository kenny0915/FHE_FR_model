#!/bin/bash
#SBATCH --account=MST114196
#SBATCH --job-name=r50-p2-conflict
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --output=work_dirs/ms1mv3_r50_herpn_full_conversion_phase2_conflict_aware_recovery/slurm-%j.out
#SBATCH --error=work_dirs/ms1mv3_r50_herpn_full_conversion_phase2_conflict_aware_recovery/slurm-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=910915kenny@gmail.com

set -euo pipefail

output=work_dirs/ms1mv3_r50_herpn_full_conversion_phase2_conflict_aware_recovery
mkdir -p "${output}"

ml load miniconda3
conda activate face_recog

export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0,1,2,3

torchrun \
    --master_addr=127.0.0.1 \
    --master_port=$((37000 + SLURM_JOB_ID % 20000)) \
    --nproc_per_node=4 \
    train_tail_recovery.py \
    configs/ms1mv3_r50_no_relu_phase2_conflict_aware_recovery

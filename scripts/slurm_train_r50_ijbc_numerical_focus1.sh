#!/bin/bash
#SBATCH --account=MST114196
#SBATCH --job-name=r50-ijbc-focus1
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --output=work_dirs/ms1mv3_r50_herpn_full_conversion_phase2_ijbc_focus1/slurm-%j.out
#SBATCH --error=work_dirs/ms1mv3_r50_herpn_full_conversion_phase2_ijbc_focus1/slurm-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=910915kenny@gmail.com

set -euo pipefail
cd /work/u8798807/FHE_FR_model
output=work_dirs/ms1mv3_r50_herpn_full_conversion_phase2_ijbc_focus1
mkdir -p "${output}"
ml load miniconda3
conda activate face_recog
export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun --master_addr=127.0.0.1 \
    --master_port=$((43000 + SLURM_JOB_ID % 20000)) --nproc_per_node=4 \
    train_ijbc_numerical_calibration.py \
    configs/ms1mv3_r50_no_relu_phase2_ijbc_numerical_focus1

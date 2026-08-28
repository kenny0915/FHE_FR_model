#!/bin/bash
#SBATCH --account=MST114196
#SBATCH --job-name=r50-nl13-herpn
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:8
#SBATCH --time=2-00:00:00
#SBATCH --output=work_dirs/ms1mv3_r50_nl13_prelu_herpn/resume-%j.out
#SBATCH --error=work_dirs/ms1mv3_r50_nl13_prelu_herpn/resume-%j.err

set -euo pipefail

cd /work/u8798807/FHE_FR_model

# Resume with four ranks because PartialFC checkpoints are sharded across the
# four ranks of the interrupted run. The allocation exposes eight H200s, but
# changing world size while restoring rank-local classifier state is unsafe.
export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=8

/home/u8798807/.conda/envs/face_recog/bin/torchrun \
    --standalone \
    --nproc_per_node=4 \
    train_v2.py configs/ms1mv3_r50_nl13_prelu_herpn_resume

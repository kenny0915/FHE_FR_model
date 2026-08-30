#!/bin/bash
#SBATCH --account=MST114196
#SBATCH --job-name=ijbc-nl13-herpn
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --output=work_dirs/ms1mv3_r50_nl13_prelu_herpn/ijbc-%j.out
#SBATCH --error=work_dirs/ms1mv3_r50_nl13_prelu_herpn/ijbc-%j.err

set -euo pipefail

cd /work/u8798807/FHE_FR_model

checkpoint="${1:-work_dirs/ms1mv3_r50_nl13_prelu_herpn/model.pt}"
tag="${2:-latest}"
output_root="${3:-work_dirs/ms1mv3_r50_nl13_prelu_herpn}"
network="${4:-r50_nl13_prelu_herpn}"
result_dir="${output_root}/ijbc_${tag}"

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=12
export PYTHONUNBUFFERED=1

/home/u8798807/.conda/envs/face_recog/bin/python eval_ijbc.py \
    --model-prefix "${checkpoint}" \
    --image-path ijb/IJBC \
    --result-dir "${result_dir}" \
    --batch-size 128 \
    --job "nl13_herpn_${tag}" \
    --target IJBC \
    --network "${network}" \
    --fail-on-nonfinite

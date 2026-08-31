#!/bin/bash
#SBATCH --account=MST114196
#SBATCH --job-name=ijbc-r50-e10-linear9
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:8
#SBATCH --exclude=25a-hgpn[143-145]
#SBATCH --mem=300G
#SBATCH --time=02:00:00
#SBATCH --output=work_dirs/ms1mv3_r50_herpn_epoch10_linear9_layer42/ijbc-%j.out
#SBATCH --error=work_dirs/ms1mv3_r50_herpn_epoch10_linear9_layer42/ijbc-%j.err

set -euo pipefail

cd /work/u8798807/FHE_FR_model

checkpoint="${1:-work_dirs/ms1mv3_r50_herpn_epoch10_linear9_layer42/model_linear9_prelu_slope_static.pt}"
tag="${2:-prelu_slope_static}"
output_root="${3:-work_dirs/ms1mv3_r50_herpn_epoch10_linear9_layer42}"
linear_index="${4:-24}"
result_dir="${output_root}/ijbc_${tag}"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=48
export PYTHONUNBUFFERED=1

model_kwargs="{\"num_features\":512,\"herpn_progress\":0.0,\"herpn_bn_eps\":0.0001,\"herpn_range_limit\":1.0,\"prelu_herpn_layerwise_scale\":true,\"prelu_herpn_initial_scale\":1.0,\"prelu_herpn_distill_eps\":0.0001,\"prelu_herpn_legacy_prefix\":8,\"prelu_herpn_linear_indices\":[${linear_index}]}"

/home/u8798807/.conda/envs/face_recog/bin/python eval_ijbc.py \
    --model-prefix "${checkpoint}" \
    --image-path ijb/IJBC \
    --result-dir "${result_dir}" \
    --batch-size 1024 \
    --job "r50_e10_linear9_${tag}" \
    --target IJBC \
    --network r50_prelu_herpn \
    --model-kwargs "${model_kwargs}" \
    --fail-on-nonfinite

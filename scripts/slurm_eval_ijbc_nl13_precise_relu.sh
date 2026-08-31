#!/bin/bash
#SBATCH --account=MST114196
#SBATCH --job-name=ijbc-nl13-poly
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --exclude=25a-hgpn[143-145]
#SBATCH --mem=200G
#SBATCH --time=4:00:00
#SBATCH --output=work_dirs/slurm-%x-%j.out
#SBATCH --error=work_dirs/slurm-%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=910915kenny@gmail.com

set -euo pipefail

cd /work/u8798807/FHE_FR_model

if [[ $# -ne 5 ]]; then
    echo "usage: sbatch $0 CHECKPOINT TAG OUTPUT DEGREE SCALE" >&2
    exit 2
fi

checkpoint="$1"
tag="$2"
output_root="$3"
degree="$4"
scale="$5"
result_dir="${output_root}/ijbc_${tag}"

if [[ "${degree}" == "4" ]]; then
    lower_degrees='[16,8,4]'
elif [[ "${degree}" == "8" ]]; then
    lower_degrees='[16,8]'
else
    echo "unsupported final degree: ${degree}" >&2
    exit 2
fi

model_kwargs=$(printf \
    '{"arch_config":"nl13","precise_relu_input_scale":%s,"precise_relu_lower_degrees":%s}' \
    "${scale}" "${lower_degrees}")

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=12
export PYTHONUNBUFFERED=1

/home/u8798807/.conda/envs/face_recog/bin/python eval_ijbc.py \
    --model-prefix "${checkpoint}" \
    --image-path ijb/IJBC \
    --result-dir "${result_dir}" \
    --batch-size 512 \
    --job "nl13_poly_${tag}" \
    --target IJBC \
    --network r50_precise_relu \
    --model-kwargs "${model_kwargs}" \
    --fail-on-nonfinite

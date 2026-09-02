#!/bin/bash
#SBATCH --account=MST114196
#SBATCH --job-name=ijbc-r50-joint
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --exclude=25a-hgpn[143-145]
#SBATCH --mem=200G
#SBATCH --time=4:00:00
#SBATCH --output=work_dirs/ms1mv3_r50_herpn_full_conversion_phase2_joint_recovery/ijbc-%j.out
#SBATCH --error=work_dirs/ms1mv3_r50_herpn_full_conversion_phase2_joint_recovery/ijbc-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=910915kenny@gmail.com

set -euo pipefail

cd /work/u8798807/FHE_FR_model

if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo "usage: sbatch $0 CHECKPOINT TAG [OUTPUT_ROOT]" >&2
    exit 2
fi

checkpoint="$1"
tag="$2"
output_root="${3:-work_dirs/ms1mv3_r50_herpn_full_conversion_phase2_joint_recovery}"
result_dir="${output_root}/ijbc_${tag}"

if [[ ! -f "${checkpoint}" ]]; then
    echo "checkpoint does not exist: ${checkpoint}" >&2
    exit 2
fi

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=12
export PYTHONUNBUFFERED=1

mkdir -p "${result_dir}"
/home/u8798807/.conda/envs/face_recog/bin/python eval_ijbc.py \
    --model-prefix "${checkpoint}" \
    --image-path ijb/IJBC \
    --result-dir "${result_dir}" \
    --batch-size 512 \
    --job "r50_joint_${tag}" \
    --target IJBC \
    --network r50_no_relu \
    --nonfinite-manifest "${result_dir}/nonfinite_manifest.csv" \
    --model-kwargs '{"num_features":512,"herpn_range_limit":6.0,"herpn_bn_eps":0.0001,"herpn_progress":5.0,"herpn_independent_basis_scales":true}'

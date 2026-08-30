#!/bin/bash
#SBATCH --account=MST114196
#SBATCH --job-name=ijbc-pillar-s4-exact
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --exclude=25a-hgpn[143-144]
#SBATCH --cpus-per-task=12
#SBATCH --mem=200G
#SBATCH --time=04:00:00
#SBATCH --output=work_dirs/ms1mv3_r50_pillar_espn_d4_scale4_unclipped/ijbc-%j.out
#SBATCH --error=work_dirs/ms1mv3_r50_pillar_espn_d4_scale4_unclipped/ijbc-%j.err

set -euo pipefail

cd /work/u8798807/FHE_FR_model

checkpoint="${1:-work_dirs/ms1mv3_r50_pillar_espn_d4_scale4_unclipped/model.pt}"
tag="${2:-final}"
result_dir="work_dirs/ms1mv3_r50_pillar_espn_d4_scale4_unclipped/ijbc_${tag}"

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=12

/home/u8798807/.conda/envs/face_recog/bin/python eval_ijbc.py \
  --model-prefix "${checkpoint}" \
  --image-path ijb/IJBC \
  --result-dir "${result_dir}" \
  --batch-size 256 \
  --job "pillar_espn_scale4_unclipped_${tag}" \
  --target IJBC \
  --network r50_pillar \
  --fail-on-nonfinite

#!/usr/bin/env bash
#SBATCH --account=MST114196
#SBATCH --job-name=controlled-d2-canary
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
set -euo pipefail
cd /work/u8798807/FHE_FR_model
: "${CHECKPOINT:?Set CHECKPOINT}"
: "${CANARY_ROOT:?Set CANARY_ROOT}"
: "${CANARY_OUTPUT:?Set CANARY_OUTPUT}"
mkdir -p "$(dirname "${CANARY_OUTPUT}")"
ml load miniconda3
conda activate face_recog
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8
python -m controlled_degree2.eval_canary \
  --checkpoint "${CHECKPOINT}" \
  --canary-root "${CANARY_ROOT}" \
  --sets "${CANARY_SETS:-lfw}" \
  --batch-size "${CANARY_BATCH_SIZE:-256}" \
  --output "${CANARY_OUTPUT}"

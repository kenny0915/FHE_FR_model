#!/usr/bin/env bash
#SBATCH --account=MST114196
#SBATCH --job-name=controlled-d2-mine
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --time=3:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
set -euo pipefail
cd /work/u8798807/FHE_FR_model
: "${CHECKPOINT:?Set CHECKPOINT to a controlled degree-2 checkpoint}"
: "${DATASET_ROOT:?Set DATASET_ROOT to MS1MV3}"
: "${OUTPUT_MANIFEST:?Set OUTPUT_MANIFEST}"
mkdir -p "$(dirname "${OUTPUT_MANIFEST}")"
ml load miniconda3
conda activate face_recog
export OMP_NUM_THREADS=8
torchrun --standalone --nproc_per_node=4 \
  -m controlled_degree2.mine_deployment_tails \
  --checkpoint "${CHECKPOINT}" \
  --dataset-root "${DATASET_ROOT}" \
  --output "${OUTPUT_MANIFEST}" \
  --batch-size "${MINE_BATCH_SIZE:-512}" \
  --workers "${MINE_WORKERS:-4}" \
  --topk "${MINE_TOPK:-64}" \
  --progress-batches "${MINE_PROGRESS_BATCHES:-100}" \
  --limit-batches "${MINE_LIMIT_BATCHES:-0}" \
  --both-orientations

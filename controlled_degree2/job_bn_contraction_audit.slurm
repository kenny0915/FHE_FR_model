#!/usr/bin/env bash
#SBATCH --account=MST114196
#SBATCH --job-name=controlled-d2-bn-audit
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
: "${DATASET_ROOT:?Set DATASET_ROOT to MS1MV3}"
: "${TAIL_MANIFEST:?Set TAIL_MANIFEST to an MS1MV3 mining manifest}"
: "${AUDIT_OUTPUT:?Set AUDIT_OUTPUT}"
mkdir -p "$(dirname "${AUDIT_OUTPUT}")"
ml load miniconda3
conda activate face_recog
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8
candidate_args=()
for candidate in ${CONTRACTION_CANDIDATES:-1.0 0.98 0.95 0.9 0.85 0.8}; do
  candidate_args+=(--candidate "${candidate}")
done
scope_args=()
for scope in ${CONTRACTION_SCOPES:-layer1.1.prelu}; do
  scope_args+=(--scope "${scope}")
done
python -m controlled_degree2.audit_bn_contraction \
  --checkpoint "${CHECKPOINT}" \
  --dataset-root "${DATASET_ROOT}" \
  --manifest "${TAIL_MANIFEST}" \
  --manifest-key "${TAIL_MANIFEST_KEY:-output_nonfinite}" \
  --output "${AUDIT_OUTPUT}" \
  --workers "${AUDIT_WORKERS:-0}" \
  "${scope_args[@]}" "${candidate_args[@]}"

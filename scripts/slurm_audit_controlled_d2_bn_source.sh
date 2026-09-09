#!/bin/bash
#SBATCH --account=MST114196
#SBATCH --job-name=d2-bn-audit
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --output=work_dirs/controlled-degree2-bn-audit-%j.out
#SBATCH --error=work_dirs/controlled-degree2-bn-audit-%j.err

set -euo pipefail
cd /work/u8798807/FHE_FR_model

: "${CHECKPOINT:?export CHECKPOINT as a controlled degree-2 checkpoint}"
: "${DATASET_TYPE:?export DATASET_TYPE as ms1mv3, wider, or ytf}"
: "${DATASET_ROOT:?export DATASET_ROOT as the source dataset root}"
: "${MANIFEST:?export MANIFEST as the matching mining manifest}"
: "${OUTPUT:?export OUTPUT as the audit JSON destination}"

manifest_key=${MANIFEST_KEY:-combined_orientations}
stress_variants=${STRESS_VARIANTS:-base}
wider_context_scales=${WIDER_CONTEXT_SCALES:-1.0,1.5}
scope=${SCOPE:-layer1.1.prelu}
candidate_spec=${CANDIDATES:-1+0.98+0.97+0.96+0.95+0.94+0.92}

IFS=+ read -r -a candidates <<< "${candidate_spec}"
candidate_args=()
for candidate in "${candidates[@]}"; do
    candidate_args+=(--candidate "${candidate}")
done

optional_args=()
if [[ -n "${ANNOTATIONS:-}" ]]; then
    optional_args+=(--annotations "${ANNOTATIONS}")
fi

ml load miniconda3
conda activate face_recog
export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0

python -m controlled_degree2.audit_bn_contraction \
    --checkpoint "${CHECKPOINT}" \
    --dataset-type "${DATASET_TYPE}" \
    --dataset-root "${DATASET_ROOT}" \
    --manifest "${MANIFEST}" \
    --manifest-key "${manifest_key}" \
    --stress-variants "${stress_variants}" \
    --wider-context-scales "${wider_context_scales}" \
    --scope "${scope}" \
    --output "${OUTPUT}" \
    --batch-size 256 \
    --workers 4 \
    "${candidate_args[@]}" \
    "${optional_args[@]}"

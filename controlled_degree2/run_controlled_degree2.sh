#!/usr/bin/env bash
set -euo pipefail

# Required inputs.  RUN10_CKPT is read only; nothing under poly_run10 is edited.
: "${TEACHER_CKPT:?Set TEACHER_CKPT to the confirmed work_dirs ResNet-50 checkpoint}"
: "${RUN10_CKPT:?Set RUN10_CKPT to run10 student_best.pt}"
: "${DATASET_ROOT:?Set DATASET_ROOT to the MS1MV3 directory with train.rec/train.idx}"

# MXNet 1.9 and the repository's torch 2.1 build require the pinned NumPy 1.x
# environment.  Fail before reserving GPU-hours if that environment has drifted.
python -c 'import numpy as np, torch; assert int(np.__version__.split(".")[0]) < 2, "recreate the pinned face_recog environment (NumPy must be <2)"; torch.from_numpy(np.zeros(1, dtype=np.float32))'

OUTPUT_ROOT="${OUTPUT_ROOT:-work_dirs/controlled_direct_degree2}"
CANARY_ROOT="${CANARY_ROOT:-${DATASET_ROOT}}"
CANARY_SETS="${CANARY_SETS:-lfw}"
GPUS="${GPUS:-4}"
STAGE="${1:-all}"

mkdir -p "${OUTPUT_ROOT}"

if [[ "${STAGE}" == "calibrate" || "${STAGE}" == "all" ]]; then
  python -m controlled_degree2.calibrate \
    --teacher "${TEACHER_CKPT}" \
    --reference-run10 "${RUN10_CKPT}" \
    --dataset-root "${DATASET_ROOT}" \
    --num-images 100000 \
    --out "${OUTPUT_ROOT}/degree2_calibration.json"
fi

if [[ "${STAGE}" == "convert" || "${STAGE}" == "all" ]]; then
  python -m controlled_degree2.convert_teacher \
    --teacher "${TEACHER_CKPT}" \
    --calibration "${OUTPUT_ROOT}/degree2_calibration.json" \
    --dataset-root "${DATASET_ROOT}" \
    --out "${OUTPUT_ROOT}/student_init.pt"
fi

if [[ "${STAGE}" == "train" || "${STAGE}" == "all" ]]; then
  # Stage A: teacher start, layer-by-layer replacement.  On four GPUs,
  # accumulate four 128-image microbatches to preserve global batch 2048.
  torchrun --standalone --nproc_per_node="${GPUS}" -m controlled_degree2.train \
    --student-init "${OUTPUT_ROOT}/student_init.pt" \
    --teacher "${TEACHER_CKPT}" \
    --dataset-root "${DATASET_ROOT}" \
    --canary-root "${CANARY_ROOT}" \
    --canary-sets "${CANARY_SETS}" \
    --output-dir "${OUTPUT_ROOT}/progressive" \
    --epochs 8 \
    --batch-size 128 \
    --global-batch 2048 \
    --swap-epochs 2 \
    --penalty-warmup-epochs 1.0 \
    --lam-reg-ratio 0.6 \
    --beta 1.0 \
    --hint-start 1.0 \
    --hint-end 0.3 \
    --aug-crop 0.1 \
    --aug-lowres 0.2 \
    --aug-photo 0.2 \
    --aug-stress 0.4 \
    --aug-pathological 0.05

  # Stage B: the settled run10 three-epoch all-polynomial polish recipe.
  torchrun --standalone --nproc_per_node="${GPUS}" -m controlled_degree2.train \
    --student-init "${OUTPUT_ROOT}/progressive/student_final.pt" \
    --teacher "${TEACHER_CKPT}" \
    --dataset-root "${DATASET_ROOT}" \
    --canary-root "${CANARY_ROOT}" \
    --canary-sets "${CANARY_SETS}" \
    --output-dir "${OUTPUT_ROOT}/polish" \
    --epochs 3 \
    --batch-size 128 \
    --global-batch 2048 \
    --swap-epochs 0 \
    --penalty-warmup-epochs 0.1 \
    --lam-reg-ratio 0.6 \
    --beta 1.0 \
    --hint-start 1.0 \
    --hint-end 0.3 \
    --aug-crop 0.1 \
    --aug-lowres 0.2 \
    --aug-photo 0.2 \
    --aug-stress 0.4 \
    --aug-pathological 0.05
fi

if [[ "${STAGE}" != "calibrate" && "${STAGE}" != "convert" && "${STAGE}" != "train" && "${STAGE}" != "all" ]]; then
  echo "usage: $0 [calibrate|convert|train|all]" >&2
  exit 2
fi

#!/usr/bin/env bash
set -euo pipefail

# MS1MV3-only refinement from a completed controlled-degree2 checkpoint.
# This deliberately has no IJB-C inputs: replay is mined online from the
# training batches and the stress transforms are synthetic/non-IJB.
: "${STUDENT_INIT:?Set STUDENT_INIT to a controlled degree-2 checkpoint}"
: "${TEACHER_CKPT:?Set TEACHER_CKPT to the teacher checkpoint}"
: "${DATASET_ROOT:?Set DATASET_ROOT to MS1MV3}"
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT for this refinement}"

python -c 'import numpy as np, torch; assert int(np.__version__.split(".")[0]) < 2; torch.from_numpy(np.zeros(1, dtype=np.float32))'

GPUS="${GPUS:-4}"
FREEZE_ARGS=()
if [[ "${FREEZE_THROUGH_LAYER3:-0}" == "1" ]]; then
  FREEZE_ARGS+=(--freeze-through-layer3)
fi
if [[ "${FREEZE_THROUGH_LAYER4:-0}" == "1" ]]; then
  FREEZE_ARGS+=(--freeze-through-layer4)
fi
DEPLOYMENT_TAIL_ARGS=()
if [[ -n "${DEPLOYMENT_TAIL_MANIFEST:-}" ]]; then
  DEPLOYMENT_TAIL_ARGS+=(
    --deployment-tail-manifest "${DEPLOYMENT_TAIL_MANIFEST}"
    --deployment-tail-batch-size "${DEPLOYMENT_TAIL_BATCH_SIZE:-8}"
    --deployment-tail-workers "${DEPLOYMENT_TAIL_WORKERS:-2}"
    --deployment-tail-beta "${DEPLOYMENT_TAIL_BETA:-0.25}"
    --deployment-tail-guard-ratio "${DEPLOYMENT_TAIL_GUARD_RATIO:-0.8}"
    --deployment-tail-priority-count "${DEPLOYMENT_TAIL_PRIORITY_COUNT:-0}"
    --deployment-tail-priority-repeats "${DEPLOYMENT_TAIL_PRIORITY_REPEATS:-1}"
  )
fi
torchrun --standalone --nproc_per_node="${GPUS}" -m controlled_degree2.train \
  --student-init "${STUDENT_INIT}" \
  --teacher "${TEACHER_CKPT}" \
  --dataset-root "${DATASET_ROOT}" \
  --canary-root "${CANARY_ROOT:-${DATASET_ROOT}}" \
  --canary-sets "${CANARY_SETS:-lfw}" \
  --output-dir "${OUTPUT_ROOT}" \
  --epochs "${EPOCHS:-3}" --batch-size 128 --global-batch 2048 \
  --lr-at-512 "${LR_AT_512:-5e-4}" \
  --swap-epochs 0 --penalty-warmup-epochs 0.1 \
  --w-embedding "${W_EMBEDDING:-1.0}" \
  --lam-reg-ratio "${LAM_REG_RATIO:-0.6}" --beta "${RANGE_BETA:-1.0}" \
  --hint-start "${HINT_START:-1.0}" --hint-end "${HINT_END:-0.3}" \
  --aug-crop "${AUG_CROP:-0.1}" --aug-lowres "${AUG_LOWRES:-0.25}" \
  --aug-photo "${AUG_PHOTO:-0.25}" \
  --aug-stress "${AUG_STRESS:-0.6}" --aug-pathological "${AUG_PATHOLOGICAL:-0.1}" \
  --tail-replay-fraction "${TAIL_REPLAY_FRACTION:-0.5}" \
  --tail-replay-capacity "${TAIL_REPLAY_CAPACITY:-4096}" \
  --tail-replay-warmup-steps 100 --causal-tail-beta "${CAUSAL_TAIL_BETA:-2.0}" \
  --activation-guard-ratio "${ACTIVATION_GUARD_RATIO:-0.5}" \
  --activation-lam-scale 1.0 \
  --activation-lam-scale-layer3 "${ACTIVATION_LAM_SCALE_LAYER3:-1.25}" \
  --operator-bound-weight "${OPERATOR_BOUND_WEIGHT:-1e-3}" \
  --operator-bound-margin "${OPERATOR_BOUND_MARGIN:-0.05}" \
  --adversarial-tail-fraction "${ADVERSARIAL_TAIL_FRACTION:-0}" \
  --adversarial-tail-steps "${ADVERSARIAL_TAIL_STEPS:-0}" \
  --adversarial-tail-epsilon "${ADVERSARIAL_TAIL_EPSILON:-0.25}" \
  --adversarial-tail-step-size "${ADVERSARIAL_TAIL_STEP_SIZE:-0.125}" \
  --adversarial-tail-beta "${ADVERSARIAL_TAIL_BETA:-1}" \
  --adversarial-tail-warmup-steps "${ADVERSARIAL_TAIL_WARMUP_STEPS:-0}" \
  --log-every "${LOG_EVERY:-50}" \
  --limit-batches "${LIMIT_BATCHES:-0}" \
  "${DEPLOYMENT_TAIL_ARGS[@]}" \
  "${FREEZE_ARGS[@]}"

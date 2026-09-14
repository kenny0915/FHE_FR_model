#!/usr/bin/env bash
#SBATCH --account=MST114196
#SBATCH --job-name=ijbc-controlled-d2
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH --time=4:00:00
set -euo pipefail
cd /work/u8798807/FHE_FR_model
: "${CHECKPOINT:?Set CHECKPOINT}"; : "${RESULT_DIR:?Set RESULT_DIR}"
test -f "${CHECKPOINT}"
if [[ "${CHANNELWISE_ACCEPTANCE:-0}" == 1 ]]; then
  [[ "${NETWORK:-r50_controlled_d2}" == r50_controlled_d2 ]]
  [[ "${POLYNOMIAL_EXPORT:-0}" == 1 && "${FINITE_AUDIT:-0}" == 1 ]]
  [[ "${POLYNOMIAL_COEFFICIENT_MODE:-shared}" == channelwise ]]
  # Freeze the evaluated weights before either the full or remainder model
  # loads them. A training job must not be able to change a live candidate.
  mkdir "${RESULT_DIR}"
  cp --reflink=auto "${CHECKPOINT}" "${RESULT_DIR}/evaluated_checkpoint.pt"
  CHECKPOINT="${RESULT_DIR}/evaluated_checkpoint.pt"
else
  mkdir -p "${RESULT_DIR}"
fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" OMP_NUM_THREADS=12 PYTHONUNBUFFERED=1
audit_args=()
if [[ "${POLYNOMIAL_EXPORT:-0}" == 1 ]]; then
  audit_args+=(--polynomial-export "${RESULT_DIR}/polynomial_certificate.json"
               --polynomial-coefficient-mode "${POLYNOMIAL_COEFFICIENT_MODE:-shared}")
fi
if [[ "${FINITE_AUDIT:-0}" == 1 ]]; then
  audit_args+=(--finite-audit "${RESULT_DIR}/finite_audit.json")
fi
/home/u8798807/.conda/envs/face_recog/bin/python eval_ijbc.py \
  --model-prefix "${CHECKPOINT}" --image-path ijb/IJBC \
  --result-dir "${RESULT_DIR}" --batch-size 512 \
  --job "${EVAL_JOB_NAME:-controlled_d2}" --target IJBC \
  --network "${NETWORK:-r50_controlled_d2}" \
  --nonfinite-manifest "${RESULT_DIR}/nonfinite_manifest.csv" "${audit_args[@]}"
if [[ "${CHANNELWISE_ACCEPTANCE:-0}" == 1 ]]; then
  /home/u8798807/.conda/envs/face_recog/bin/python -m controlled_degree2.channelwise_acceptance \
    --raw "${RESULT_DIR}/${EVAL_JOB_NAME:-controlled_d2}/ijbc_tar_at_far_raw.json" \
    --audit "${RESULT_DIR}/finite_audit.json" \
    --certificate "${RESULT_DIR}/polynomial_certificate.json" \
    --checkpoint "${CHECKPOINT}" --output "${RESULT_DIR}/acceptance.json"
fi

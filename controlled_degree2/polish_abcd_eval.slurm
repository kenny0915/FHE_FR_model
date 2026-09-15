#!/usr/bin/env bash
#SBATCH --account=MST114196
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:H200:1
#SBATCH --mem=160G
#SBATCH --time=04:00:00
set -euo pipefail
cd "${ABCD_CODE_ROOT:-${SLURM_SUBMIT_DIR:?}}"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MXNET_CPU_WORKER_NTHREADS=1 PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
PY=/home/u8798807/.conda/envs/face_recog/bin/python
: "${ABCD_OUTPUT:?}"
if [[ ${ABCD_MODE:-eval} == baseline ]]; then
  CHECKPOINT="$ABCD_OUTPUT/inputs/progressive_final.pt"
  EVAL_ROOT="$ABCD_OUTPUT/baseline"
else
  arms=(A B C D)
  arm=${arms[${SLURM_ARRAY_TASK_ID:?}]}
  CHECKPOINT="$ABCD_OUTPUT/$arm/epoch2.pt"
  EVAL_ROOT="$ABCD_OUTPUT/$arm/final_eval"
fi
mkdir -p "$EVAL_ROOT"
# MS1MV3 full original/flip audit: recorded independently of IJB-C.
"$PY" -m controlled_degree2.mine_deployment_tails --checkpoint "$CHECKPOINT" \
  --dataset-root ms1m-retinaface-t1 --output "$EVAL_ROOT/ms1mv3/manifest.json" \
  --batch-size 256 --workers 4 --topk 32
export CHECKPOINT RESULT_DIR="$EVAL_ROOT/ijbc" EVAL_JOB_NAME=polish_abcd
export POLYNOMIAL_EXPORT=1 FINITE_AUDIT=1 POLYNOMIAL_COEFFICIENT_MODE=channelwise
# Do not require the unrelated historical TAR>=96% acceptance target.
mkdir -p "$RESULT_DIR"
"$PY" eval_ijbc.py --model-prefix "$CHECKPOINT" --image-path ijb/IJBC \
  --result-dir "$RESULT_DIR" --batch-size 512 --job polish_abcd --target IJBC \
  --network r50_controlled_d2 --nonfinite-manifest "$RESULT_DIR/nonfinite_manifest.csv" \
  --polynomial-export "$RESULT_DIR/polynomial_certificate.json" \
  --polynomial-coefficient-mode channelwise --finite-audit "$RESULT_DIR/finite_audit.json"
"$PY" -m controlled_degree2.polish_abcd_result --root "$EVAL_ROOT"

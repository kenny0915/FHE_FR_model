#!/usr/bin/env bash
#SBATCH --account=MST114196
#SBATCH --job-name=d2-accuracy-ijbc
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --array=0-2%3
#SBATCH --time=04:00:00
set -euo pipefail
cd "${ACCURACY_CODE_ROOT:-${SLURM_SUBMIT_DIR:?}}"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
ACCURACY_PYTHON="${ACCURACY_PYTHON:-${HOME}/.conda/envs/face_recog/bin/python}"
"$ACCURACY_PYTHON" -c 'import sys,numpy,torch; print("runtime:",sys.executable,"numpy",numpy.__version__,"torch",torch.__version__,flush=True); assert torch.cuda.device_count()==1, "one visible GPU required"'
arms=(control fixed coefficients)
arm="${arms[${SLURM_ARRAY_TASK_ID:?}]}"
: "${ACCURACY_EVAL_INPUTS:?}"
: "${ACCURACY_EVAL_OUTPUT:?}"
: "${IJBC_ROOT:?}"
result="${ACCURACY_EVAL_OUTPUT}/${arm}"
# Refuse to mix artifacts from different evaluations.
mkdir "$result"
"$ACCURACY_PYTHON" eval_ijbc.py \
  --model-prefix "${ACCURACY_EVAL_INPUTS}/${arm}.pt" \
  --image-path "$IJBC_ROOT" --result-dir "$result" \
  --batch-size 256 --image-workers 8 --job "${arm}_epoch3_nonfinite_only" \
  --target IJBC --network r50_controlled_d2 \
  --bts-failure-report "${result}/failures" --ignore-bts-range

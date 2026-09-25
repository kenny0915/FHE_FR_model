#!/usr/bin/env bash
#SBATCH --account=MST114196
#SBATCH --job-name=control-bts-ijbc
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --array=0-4%5
#SBATCH --time=04:00:00
set -euo pipefail
cd "${BTS_CODE_ROOT:-${SLURM_SUBMIT_DIR:?}}"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
PYTHON_BIN="${ACCURACY_PYTHON:-${HOME}/.conda/envs/face_recog/bin/python}"
"$PYTHON_BIN" -c 'import sys,numpy,torch; print("runtime:",sys.executable,"numpy",numpy.__version__,"torch",torch.__version__,flush=True); assert torch.cuda.device_count()==1, "one visible GPU required"'
layouts=(bts6 bts14 bts9 bts7 bts5)
layout="${layouts[${SLURM_ARRAY_TASK_ID:?}]}"
: "${BTS_EVAL_INPUTS:?}"
: "${BTS_EVAL_OUTPUT:?}"
: "${IJBC_ROOT:?}"
result="${BTS_EVAL_OUTPUT}/${layout}"
mkdir "$result"
# Metadata selects the exact BTS locations; enforce inclusive [-1,1].
"$PYTHON_BIN" eval_ijbc.py \
  --model-prefix "${BTS_EVAL_INPUTS}/${layout}.pt" \
  --image-path "$IJBC_ROOT" --result-dir "$result" \
  --batch-size 256 --image-workers 8 --job "${layout}_zero_failed" \
  --target IJBC --network r50_controlled_d2 \
  --bts-failure-report "${result}/failures"

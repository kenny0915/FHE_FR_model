#!/usr/bin/env bash
#SBATCH --account=MST114196
#SBATCH --job-name=d2-relation
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --mem=160G
#SBATCH --array=0-2%3
#SBATCH --time=06:00:00
set -euo pipefail
cd "${ACCURACY_CODE_ROOT:-${SLURM_SUBMIT_DIR:?}}"
source controlled_degree2/accuracy_runtime.sh
smoke_args=()
case "${ACCURACY_MODE:-train}" in
  train) ;;
  smoke) smoke_args=(--smoke) ;;
  *) echo 'ACCURACY_MODE must be train or smoke' >&2; exit 1 ;;
esac
arms=(pointwise relation01 relation1)
"$ACCURACY_PYTHON" -m controlled_degree2.relational_recovery \
  --arm "${arms[${SLURM_ARRAY_TASK_ID:?}]}" --checkpoint "${CHECKPOINT:?}" \
  --teacher "${TEACHER:?}" --dataset-root "${DATASET_ROOT:?}" \
  --canary-root "${CANARY_ROOT:?}" --output-root "${OUTPUT_ROOT:?}" \
  "${smoke_args[@]}" --run

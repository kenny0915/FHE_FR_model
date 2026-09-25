#!/usr/bin/env bash
#SBATCH --account=MST114196
#SBATCH --job-name=d2-accuracy
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --array=0-2%3
#SBATCH --time=04:00:00
#SBATCH --output=d2-accuracy-%A_%a.out
#SBATCH --error=d2-accuracy-%A_%a.err
set -euo pipefail
cd "${ACCURACY_CODE_ROOT:-${SLURM_SUBMIT_DIR:?}}"
source controlled_degree2/accuracy_runtime.sh
smoke_args=()
case "${ACCURACY_MODE:-train}" in
  train) ;;
  smoke) smoke_args=(--smoke) ;;
  *) echo 'ACCURACY_MODE must be train or smoke' >&2; exit 1 ;;
esac
arms=(control fixed coefficients)
"$ACCURACY_PYTHON" -m controlled_degree2.accuracy_recovery \
  --arm "${arms[$SLURM_ARRAY_TASK_ID]}" \
  --checkpoint "${CHECKPOINT:-work_dirs/controlled_degree2_tail_ms1mv3_20260907/progressive/student_best.pt}" \
  --teacher "${TEACHER:-work_dirs/ms1mv3_r50/model.pt}" \
  --dataset-root "${DATASET_ROOT:-ms1m-retinaface-t1}" \
  --canary-root "${CANARY_ROOT:-faces_webface_112x112}" \
  --output-root "${OUTPUT_ROOT:-work_dirs/accuracy_recovery_20260925}" "${smoke_args[@]}" --run

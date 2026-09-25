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
ml load miniconda3
conda activate face_recog
cd "${ACCURACY_CODE_ROOT:-${SLURM_SUBMIT_DIR:?}}"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MXNET_CPU_WORKER_NTHREADS=1 PYTHONUNBUFFERED=1
arms=(control fixed coefficients)
python -m controlled_degree2.accuracy_recovery \
  --arm "${arms[$SLURM_ARRAY_TASK_ID]}" \
  --checkpoint "${CHECKPOINT:-work_dirs/controlled_degree2_tail_ms1mv3_20260907/progressive/student_best.pt}" \
  --teacher "${TEACHER:-work_dirs/ms1mv3_r50/model.pt}" \
  --dataset-root "${DATASET_ROOT:-ms1m-retinaface-t1}" \
  --canary-root "${CANARY_ROOT:-faces_webface_112x112}" \
  --output-root "${OUTPUT_ROOT:-work_dirs/accuracy_recovery_20260925}" --run

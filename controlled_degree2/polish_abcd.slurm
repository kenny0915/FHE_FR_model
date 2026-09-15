#!/usr/bin/env bash
#SBATCH --account=MST114196
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:H200:4
#SBATCH --mem=160G
#SBATCH --time=04:00:00
set -euo pipefail
cd "${ABCD_CODE_ROOT:-${SLURM_SUBMIT_DIR:?}}"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MXNET_CPU_WORKER_NTHREADS=1 PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
PY=/home/u8798807/.conda/envs/face_recog/bin/python
: "${ABCD_OUTPUT:?}"
if [[ ${ABCD_MODE:-train} == smoke ]]; then
  for arm in A B C D; do
    "$PY" -m torch.distributed.run --standalone --nproc_per_node=4 -m controlled_degree2.polish_abcd \
      --arm "$arm" --output "$ABCD_OUTPUT/$arm" --epochs 1 --batch-size 128 --global-batch 2048 \
      --workers 4 --audit-images 8 --smoke
  done
else
  arms=(A B C D)
  arm=${arms[${SLURM_ARRAY_TASK_ID:?}]}
  "$PY" -m torch.distributed.run --standalone --nproc_per_node=4 -m controlled_degree2.polish_abcd \
    --arm "$arm" --output "$ABCD_OUTPUT/$arm" --epochs 2 --batch-size 128 --global-batch 2048 \
    --checkpoint "$ABCD_OUTPUT/inputs/progressive_final.pt" --teacher "$ABCD_OUTPUT/inputs/teacher.pt" \
    --workers 4 --audit-images 8192 --tail-manifest "$ABCD_OUTPUT/baseline/ms1mv3/manifest.json"
fi

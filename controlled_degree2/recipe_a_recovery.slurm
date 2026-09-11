#!/usr/bin/env bash
#SBATCH --account=MST114196
#SBATCH --job-name=r50-d2-a-recovery
#SBATCH --partition=16gpus
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:H200:8
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@600
#SBATCH --open-mode=append
#SBATCH --output=recipe-a-recovery-%j.out
#SBATCH --error=recipe-a-recovery-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MXNET_CPU_WORKER_NTHREADS=1
export PYTHONUNBUFFERED=1 NCCL_DEBUG=WARN
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export RECIPE_PYTHON=/home/u8798807/.conda/envs/face_recog/bin/python
export RECIPE_MASTER_ADDR
RECIPE_MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export RECIPE_MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))
export RECIPE_OUTPUT="${RECIPE_OUTPUT:-$PWD/work_dirs/recipe_a_recovery_${SLURM_JOB_ID}}"
export RECOVERY_SOURCE="${RECOVERY_SOURCE:-$PWD/work_dirs/recipe_a_376833}"
export RECOVERY_RESUME="${RECOVERY_RESUME:-}"
export RECIPE_SMOKE="${RECIPE_SMOKE:-0}"
export RECIPE_COMMIT
RECIPE_COMMIT=$(git rev-parse HEAD)
# Only a scheduled wall-time signal requests requeue. Real code/numerical
# failures stay visible rather than entering an endless crash/restart loop.
trap 'echo "Wall-time approaching: requeue from latest atomic checkpoint"; scontrol requeue "$SLURM_JOB_ID"; exit 0' USR1
srun --nodes=2 --ntasks=2 --ntasks-per-node=1 --distribution=block:block \
    --kill-on-bad-exit=1 bash controlled_degree2/recipe_a_node.sh recovery &
wait $!

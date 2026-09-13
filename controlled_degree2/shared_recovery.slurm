#!/usr/bin/env bash
#SBATCH --account=MST114196
#SBATCH --job-name=shared-d2-gated
#SBATCH --partition=16gpus
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:H200:8
#SBATCH --time=1-00:00:00
#SBATCH --output=shared-recovery-%j.out
#SBATCH --error=shared-recovery-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
: "${RECIPE_OUTPUT:?Use a new experiment directory}"
: "${SHARED_DEADLINE:?Use the common campaign deadline}"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MXNET_CPU_WORKER_NTHREADS=1
export PYTHONUNBUFFERED=1 NCCL_DEBUG=WARN
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export RECIPE_PYTHON=/home/u8798807/.conda/envs/face_recog/bin/python
export RECIPE_MASTER_ADDR
RECIPE_MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export RECIPE_MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))
export RECOVERY_SOURCE="${RECOVERY_SOURCE:-$PWD/work_dirs/shared_d2_D_20260913}"
export SHARED_RECOVERY=1 RECIPE_SMOKE="${RECIPE_SMOKE:-0}"
export RECIPE_COMMIT
RECIPE_COMMIT=$(git rev-parse HEAD)
srun --nodes=2 --ntasks=2 --ntasks-per-node=1 --distribution=block:block \
    --kill-on-bad-exit=1 bash controlled_degree2/recipe_a_node.sh recovery-v2

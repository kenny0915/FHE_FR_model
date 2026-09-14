#!/usr/bin/env bash
#SBATCH --account=MST114196
#SBATCH --job-name=r50-channel-d2
#SBATCH --partition=16gpus
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:H200:8
#SBATCH --time=00:20:00
#SBATCH --output=channelwise-%j.out
#SBATCH --error=channelwise-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MXNET_CPU_WORKER_NTHREADS=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export CHANNEL_PYTHON=/home/u8798807/.conda/envs/face_recog/bin/python
export CHANNEL_MASTER
CHANNEL_MASTER=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export CHANNEL_PORT=$((20000 + SLURM_JOB_ID % 20000))
export CHANNEL_OUTPUT="${CHANNEL_OUTPUT:-$PWD/work_dirs/channelwise_goal_${SLURM_JOB_ID}}"
export CHANNEL_SMOKE="${CHANNEL_SMOKE:-1}"
export RECIPE_COMMIT
RECIPE_COMMIT=$(git rev-parse HEAD)
# Production always needs an explicit absolute deadline, including preparation.
if [[ "$CHANNEL_SMOKE" != 1 ]]; then
    : "${CHANNEL_DEADLINE:?Production requires an agreed absolute deadline}"
fi
for stage in prepare train; do
    srun --nodes=2 --ntasks=2 --ntasks-per-node=1 --distribution=block:block \
        --kill-on-bad-exit=1 bash controlled_degree2/channelwise_goal_node.sh "$stage"
done

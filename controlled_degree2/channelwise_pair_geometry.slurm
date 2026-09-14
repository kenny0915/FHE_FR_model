#!/usr/bin/env bash
#SBATCH --account=MST114196
#SBATCH --job-name=channelwise-pair-geometry
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:H200:1
#SBATCH --mem=200G
#SBATCH --time=00:45:00
#SBATCH --output=channelwise-%j.out
#SBATCH --error=channelwise-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
: "${CHANNEL_CACHE_ROOT:?}" "${CHANNEL_OUTPUT:?}"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 WORLD_SIZE=1
PYTHON=/home/u8798807/.conda/envs/face_recog/bin/python
geometry_recipe_commit=$(git rev-parse HEAD)
geometry_args=()
if [[ "${CHANNEL_FULL_TEMPLATE_BATCH:-0}" == 1 ]]; then
    geometry_args+=(--full-template-batch)
fi
"$PYTHON" -m controlled_degree2.calibrate_pair_geometry \
    --cache-root "$CHANNEL_CACHE_ROOT" --output "$CHANNEL_OUTPUT" \
    --steps "${CHANNEL_GEOMETRY_STEPS:-2000}" \
    --tail-threshold "${CHANNEL_TAIL_THRESHOLD:-0.3}" "${geometry_args[@]}"
printf '%s\n' "$geometry_recipe_commit" > "$CHANNEL_OUTPUT/recipe_commit.txt"
if [[ ! -f "$CHANNEL_OUTPUT/candidate.pt" ]]; then
    echo 'No held-out pair-geometry improvement; full evaluation skipped.'
    exit 0
fi
export CHECKPOINT="$CHANNEL_OUTPUT/candidate.pt" RESULT_DIR="$CHANNEL_OUTPUT/ijbc_full"
export CHANNELWISE_ACCEPTANCE=1 POLYNOMIAL_EXPORT=1 FINITE_AUDIT=1
export POLYNOMIAL_COEFFICIENT_MODE=channelwise NETWORK=r50_controlled_d2
export EVAL_JOB_NAME=pair_geometry
exec bash controlled_degree2/job_ijbc.slurm

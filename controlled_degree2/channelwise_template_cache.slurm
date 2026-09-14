#!/usr/bin/env bash
#SBATCH --account=MST114196
#SBATCH --job-name=channelwise-template-cache
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H200:1
#SBATCH --mem=100G
#SBATCH --time=00:30:00
#SBATCH --output=channelwise-%j.out
#SBATCH --error=channelwise-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
: "${CHANNEL_CHECKPOINT:?}" "${CHANNEL_OUTPUT:?}"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 WORLD_SIZE=1
template_recipe_commit=$(git rev-parse HEAD)
/home/u8798807/.conda/envs/face_recog/bin/python -m controlled_degree2.cache_template_geometry \
    --checkpoint "$CHANNEL_CHECKPOINT" --output "$CHANNEL_OUTPUT" \
    --fit-templates "${CHANNEL_FIT_TEMPLATES:-2048}" \
    --validation-templates "${CHANNEL_VALIDATION_TEMPLATES:-512}" \
    --seed "${CHANNEL_TEMPLATE_SEED:-20260927}"
printf '%s\n' "$template_recipe_commit" > "$CHANNEL_OUTPUT/recipe_commit.txt"

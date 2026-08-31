#!/bin/bash
#SBATCH --account=MST114196
#SBATCH --job-name=r50-e10-lin-screen
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:8
#SBATCH --exclude=25a-hgpn[143-145]
#SBATCH --mem=300G
#SBATCH --time=01:00:00
#SBATCH --output=work_dirs/ms1mv3_r50_herpn_epoch10_linear9_screen/slurm-%j.out
#SBATCH --error=work_dirs/ms1mv3_r50_herpn_epoch10_linear9_screen/slurm-%j.err

set -euo pipefail

cd /work/u8798807/FHE_FR_model
output=work_dirs/ms1mv3_r50_herpn_epoch10_linear9_screen
checkpoint=work_dirs/ms1mv3_r50_herpn_epoch10_linear9_layer42/model_epoch_01.pt
mkdir -p "$output"

variants=(fitted constant zero prelu_slope prelu_slope_bias small_positive_bias)
pids=()
for index in "${!variants[@]}"; do
    variant=${variants[$index]}
    CUDA_VISIBLE_DEVICES=$index \
    /home/u8798807/.conda/envs/face_recog/bin/python \
        eval/screen_linear_replacement.py \
        --checkpoint "$checkpoint" \
        --variant "$variant" \
        --batch-size 100 \
        --output "$output/$variant.json" \
        > "$output/$variant.log" 2>&1 &
    pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        failed=1
    fi
done
exit "$failed"

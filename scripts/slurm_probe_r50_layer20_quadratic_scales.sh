#!/bin/bash
#SBATCH --account=MST114196
#SBATCH --job-name=r50-l20-qprobe
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=work_dirs/ms1mv3_r50_herpn_full_conversion_phase2_quadratic_probe/slurm-%j.out
#SBATCH --error=work_dirs/ms1mv3_r50_herpn_full_conversion_phase2_quadratic_probe/slurm-%j.err

set -euo pipefail

cd /work/u8798807/FHE_FR_model
output=work_dirs/ms1mv3_r50_herpn_full_conversion_phase2_quadratic_probe
mkdir -p "${output}"

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8

trace_args=()
for name in \
    layer2.0.prelu layer2.1.prelu layer2.2.prelu layer2.3.prelu \
    layer3.0.prelu layer3.1.prelu layer3.2.prelu layer3.3.prelu; do
    trace_args+=(--trace-activation "${name}")
done

/home/u8798807/.conda/envs/face_recog/bin/python \
    probe_herpn_quadratic_scales.py configs/ms1mv3_r50_no_relu \
    --checkpoint \
      work_dirs/ms1mv3_r50_herpn_full_conversion_phase1/model_epoch_23.pt \
    --manifest \
      work_dirs/ms1mv3_r50_herpn_full_conversion_phase2_tail_mining/epoch23_prefix_tails.json \
    --activation layer2.0.prelu \
    --quadratic-scale 1.0 \
    --quadratic-scale 0.5 \
    --quadratic-scale 0.25 \
    --quadratic-scale 0.125 \
    --quadratic-scale 0.0625 \
    --quadratic-scale 0.03125 \
    --quadratic-scale 0.01 \
    --quadratic-scale 0.001 \
    --batch-size 64 \
    --workers 2 \
    --output "${output}/layer20_quadratic_scales.json" \
    --save-first-zero-checkpoint "${output}/model_first_zero_ms1m.pt" \
    "${trace_args[@]}"

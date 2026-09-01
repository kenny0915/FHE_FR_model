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
    layer3.0.prelu layer3.1.prelu layer3.2.prelu layer3.3.prelu \
    layer3.13.prelu layer4.0.prelu layer4.1.prelu layer4.2.prelu; do
    trace_args+=(--trace-activation "${name}")
done

attenuate_args=()
for stage_blocks in "layer2:4" "layer3:14"; do
    stage=${stage_blocks%%:*}
    blocks=${stage_blocks##*:}
    for index in $(seq 0 $((blocks - 1))); do
        attenuate_args+=(--activation "${stage}.${index}.prelu")
    done
done

/home/u8798807/.conda/envs/face_recog/bin/python \
    probe_herpn_quadratic_scales.py configs/ms1mv3_r50_no_relu \
    --checkpoint \
      work_dirs/ms1mv3_r50_herpn_full_conversion_phase1/model_epoch_23.pt \
    --manifest \
      work_dirs/ms1mv3_r50_herpn_full_conversion_phase2_tail_mining/epoch23_prefix_tails.json \
    --quadratic-scale 0.009 \
    --quadratic-scale 0.008 \
    --quadratic-scale 0.007 \
    --quadratic-scale 0.006 \
    --quadratic-scale 0.005 \
    --quadratic-scale 0.004 \
    --quadratic-scale 0.003 \
    --quadratic-scale 0.002 \
    --quadratic-scale 0.001 \
    --batch-size 64 \
    --workers 2 \
    --output "${output}/stage23_quadratic_scales_fine.json" \
    --save-first-zero-checkpoint "${output}/model_stage23_fine_zero_ms1m.pt" \
    "${attenuate_args[@]}" \
    "${trace_args[@]}"

#!/bin/bash
#SBATCH --account=MST114196
#SBATCH --job-name=r50-ms1m-mine25
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --output=work_dirs/ms1mv3_r50_herpn_full_conversion_phase2_ms1mv3_tail_mining_all25/slurm-%j.out
#SBATCH --error=work_dirs/ms1mv3_r50_herpn_full_conversion_phase2_ms1mv3_tail_mining_all25/slurm-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=910915kenny@gmail.com

set -euo pipefail
cd /work/u8798807/FHE_FR_model
output=work_dirs/ms1mv3_r50_herpn_full_conversion_phase2_ms1mv3_tail_mining_all25
mkdir -p "${output}"
ml load miniconda3
conda activate face_recog
export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0,1,2,3

activation_args=(--activation prelu)
for stage_blocks in "layer1:3" "layer2:4" "layer3:14" "layer4:3"; do
    stage=${stage_blocks%%:*}
    blocks=${stage_blocks##*:}
    for index in $(seq 0 $((blocks - 1))); do
        activation_args+=(--activation "${stage}.${index}.prelu")
    done
done

torchrun --master_addr=127.0.0.1 \
    --master_port=$((45000 + SLURM_JOB_ID % 20000)) --nproc_per_node=4 \
    mine_herpn_tails.py configs/ms1mv3_r50_no_relu \
    --checkpoint work_dirs/ms1mv3_r50_herpn_full_conversion_phase1/model_epoch_23.pt \
    --output "${output}/epoch23_ms1mv3_tails_all25.json" \
    --batch-size 512 --workers 4 --topk 512 --both-orientations \
    "${activation_args[@]}"

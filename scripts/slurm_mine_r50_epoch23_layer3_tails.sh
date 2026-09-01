#!/bin/bash
#SBATCH --account=MST114196
#SBATCH --job-name=r50-tail-mine
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --output=work_dirs/ms1mv3_r50_herpn_full_conversion_phase2_tail_mining/slurm-%j.out
#SBATCH --error=work_dirs/ms1mv3_r50_herpn_full_conversion_phase2_tail_mining/slurm-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=910915kenny@gmail.com

set -euo pipefail

output=work_dirs/ms1mv3_r50_herpn_full_conversion_phase2_tail_mining
mkdir -p "${output}"

ml load miniconda3
conda activate face_recog

export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0,1,2,3

activation_args=()
for index in $(seq 0 13); do
    activation_args+=(--activation "layer3.${index}.prelu")
done

torchrun \
    --master_addr=127.0.0.1 \
    --master_port=$((23000 + SLURM_JOB_ID % 20000)) \
    --nproc_per_node=4 \
    mine_herpn_tails.py configs/ms1mv3_r50_no_relu \
    --checkpoint \
      work_dirs/ms1mv3_r50_herpn_full_conversion_phase1/model_epoch_23.pt \
    --output "${output}/epoch23_layer3_tails.json" \
    --batch-size 512 \
    --workers 4 \
    --topk 512 \
    --both-orientations \
    "${activation_args[@]}"

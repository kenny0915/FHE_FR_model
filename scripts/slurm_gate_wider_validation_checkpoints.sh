#!/bin/bash
#SBATCH --account=MST114196
#SBATCH --job-name=r50-wider-val-gate
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --output=work_dirs/wider-validation-gate-%j.out
#SBATCH --error=work_dirs/wider-validation-gate-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=910915kenny@gmail.com

set -euo pipefail
cd /work/u8798807/FHE_FR_model
if [[ $# -lt 2 ]]; then
    echo "usage: sbatch $0 OUTPUT_DIR CHECKPOINT [CHECKPOINT ...]" >&2
    exit 2
fi
output_dir=$1
shift
checkpoint_args=()
for checkpoint in "$@"; do
    if [[ ! -f "${checkpoint}" ]]; then
        echo "checkpoint does not exist: ${checkpoint}" >&2
        exit 2
    fi
    checkpoint_args+=(--checkpoint "${checkpoint}")
done

mkdir -p "${output_dir}"
ml load miniconda3
conda activate face_recog
export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0,1,2,3

torchrun --master_addr=127.0.0.1 \
    --master_port=$((54000 + SLURM_JOB_ID % 11000)) --nproc_per_node=4 \
    evaluate_numerical_gate.py \
    configs/ms1mv3_r50_no_relu_phase2_wider_robust_margin \
    --dataset wider --wider-split validation \
    --range-gate-limit 4 --batch-size 256 --workers 4 \
    --output-dir "${output_dir}" "${checkpoint_args[@]}"

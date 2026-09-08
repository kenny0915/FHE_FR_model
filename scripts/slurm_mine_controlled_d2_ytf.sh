#!/bin/bash
#SBATCH --account=MST114196
#SBATCH --job-name=d2-ytf-mine
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --output=work_dirs/controlled_degree2_tail_ms1mv3_bn_contract_v14_20260908/candidate_093/ytf-mine-%j.out
#SBATCH --error=work_dirs/controlled_degree2_tail_ms1mv3_bn_contract_v14_20260908/candidate_093/ytf-mine-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=910915kenny@gmail.com

set -euo pipefail
cd /work/u8798807/FHE_FR_model

: "${YTF_ROOT:?export YTF_ROOT as the extracted aligned_images_DB directory}"

checkpoint=${CHECKPOINT:-work_dirs/controlled_degree2_tail_ms1mv3_bn_contract_v14_20260908/candidate_093/student_contracted.pt}
output_dir=${OUTPUT_DIR:-work_dirs/controlled_degree2_tail_ms1mv3_bn_contract_v14_20260908/candidate_093/ytf_mine}
stress_variants=${YTF_STRESS_VARIANTS:-base}
mkdir -p "${output_dir}"

ml load miniconda3
conda activate face_recog
export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0,1,2,3

torchrun --master_addr=127.0.0.1 \
    --master_port=$((47000 + SLURM_JOB_ID % 18000)) --nproc_per_node=4 \
    -m controlled_degree2.mine_deployment_tails \
    --checkpoint "${checkpoint}" \
    --dataset-type ytf \
    --dataset-root "${YTF_ROOT}" \
    --aligned-stress-variants "${stress_variants}" \
    --output "${output_dir}/manifest.json" \
    --batch-size 512 \
    --workers 4 \
    --topk 256 \
    --both-orientations \
    --progress-batches 50

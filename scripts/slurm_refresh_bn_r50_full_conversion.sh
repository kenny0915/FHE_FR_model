#!/bin/bash
#SBATCH --account=MST114196
#SBATCH --job-name=bn-r50-full-poly
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --exclude=25a-hgpn[143-145]
#SBATCH --mem=300G
#SBATCH --time=2:00:00
#SBATCH --output=work_dirs/ms1mv3_r50_herpn_full_conversion_phase2_bn/slurm-%j.out
#SBATCH --error=work_dirs/ms1mv3_r50_herpn_full_conversion_phase2_bn/slurm-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=910915kenny@gmail.com

set -euo pipefail

cd /work/u8798807/FHE_FR_model
mkdir -p work_dirs/ms1mv3_r50_herpn_full_conversion_phase2_bn

source_model="${1:-work_dirs/ms1mv3_r50_herpn_full_conversion_phase1/model_epoch_23.pt}"
output_model="${2:-work_dirs/ms1mv3_r50_herpn_full_conversion_phase2_bn/model_epoch23_bnreset_1000x4.pt}"

if [[ ! -f "${source_model}" ]]; then
    echo "source model does not exist: ${source_model}" >&2
    exit 2
fi

export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1

# IJB-C is not used here. Four disjoint MS1Mv3 shards contribute 1,000
# batches each (512,000 training images total). A bad batch is rejected on all
# ranks and all mutable BN buffers are rolled back before calibration resumes.
/home/u8798807/.conda/envs/face_recog/bin/torchrun \
    --standalone \
    --nproc_per_node=4 \
    refresh_bn_stats.py configs/ms1mv3_r50_no_relu_full_conversion_phase1 \
    --model "${source_model}" \
    --output "${output_model}" \
    --batch-size 128 \
    --max-batches 1000 \
    --reset-bn \
    --skip-nonfinite \
    --max-nonfinite-skips 1000 \
    --log-interval 100

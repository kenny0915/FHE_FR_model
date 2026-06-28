#!/usr/bin/env bash
set -euo pipefail

GPUS="${GPUS:-0,1,2,3}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
MASTER_PORT="${MASTER_PORT:-29500}"
CONFIG="configs/ms1mv3_poolformer_s24_mlp2"
WORK_DIR="work_dirs/ms1mv3_poolformer_s24_mlp2"

CUDA_VISIBLE_DEVICES="${GPUS}" torchrun \
  --master_port="${MASTER_PORT}" \
  --nproc_per_node="${NPROC_PER_NODE}" \
  train_v2.py "${CONFIG}"

CUDA_VISIBLE_DEVICES="${EVAL_GPU:-0}" python eval_ijbc.py \
  --model-prefix "${WORK_DIR}/model.pt" \
  --image-path "${IJB_PATH:-ijb/IJBC}" \
  --result-dir "${WORK_DIR}/ijbc_result" \
  --batch-size "${EVAL_BATCH_SIZE:-128}" \
  --job ms1mv3_poolformer_s24_mlp2 \
  --target IJBC \
  --network poolformer_s24_mlp2

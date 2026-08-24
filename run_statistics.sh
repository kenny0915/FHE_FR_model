#!/usr/bin/env bash

(
  set -euo pipefail
  statistics_repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
  cd "$statistics_repo_dir"

  torchrun eval/layer_statistics.py \
    --config configs/ms1mv3_r50.py \
    --checkpoint work_dirs/ms1mv3_r50/model.pt \
    --ijb-root ijb/IJBC \
    --target IJBC \
    --num-images 1000 \
    --batch-size 64 \
    --output work_dirs/ms1mv3_r50/ijbc_layer_statistics \
    "$@"
)

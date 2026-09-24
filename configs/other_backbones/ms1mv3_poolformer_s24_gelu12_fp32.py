"""Train the 12-GELU PoolFormer-S24 with the fully-gated FP32 recipe."""

from copy import deepcopy

from .ms1mv3_poolformer_s24_fully_gated_fp32 import config as baseline_config


config = deepcopy(baseline_config)
config.network = "poolformer_s24_gelu12"
config.arch_config = "gelu12"
config.output = "work_dirs/ms1mv3_poolformer_s24_gelu12_fp32"

# This is an explicit-activation ablation of standard PoolFormer-S24: all MLP
# ratios remain 4, GroupNorm remains in both residual branches, and exactly the
# alternating 2/2/6/2 blocks retain GELU. It intentionally reuses the FP32
# optimizer, clipping, batch-size, and schedule settings of the named baseline.

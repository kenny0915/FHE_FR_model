"""Robust fixed-graph recovery for the all-polynomial NL9 backbone.

This branch uses the same degree-2, ``[-6, 6]`` polynomial graph as the clean
recovery.  A stronger frozen-embedding constraint and bounded photometric
augmentation test whether IJB-C's rare illumination/contrast tails can be
covered without changing encrypted inference or its multiplicative depth.
"""

from easydict import EasyDict as edict

from configs.ms1mv3_r50_nl9_prelu_herpn_fixed_recovery import (
    config as _clean_config,
)


config = edict(_clean_config.copy())
config.output = (
    "work_dirs/ms1mv3_r50_nl9_prelu_herpn_fixed_recovery_aug")
config.embedding_distill_weight = 5.0
config.range_augmentation = {
    "enabled": True,
    "probability": 0.50,
    "contrast": (0.60, 1.60),
    "gain": (0.75, 1.30),
    "bias": (-0.10, 0.10),
    "gamma": (0.75, 1.35),
    "noise_probability": 0.25,
    "noise_std": 0.03,
}


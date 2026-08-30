"""Gentle true-resume hedge for selective-eight embedding recovery.

This is identical to the class-head-preserving resume experiment, but rebases
from a ``3e-4`` base LR (about ``1.09e-5`` effective backbone LR at epoch 8).
It tests whether a smaller update can gain low-FAR separation without rotating
the already finite 89.50% eight-polynomial representation.
"""

from easydict import EasyDict as edict

from configs.ms1mv3_r50_nl13_prelu_herpn_selective8_resume_distill import (
    config as _resume_config,
)


config = edict(_resume_config.copy())
config.output = (
    "work_dirs/"
    "ms1mv3_r50_nl13_prelu_herpn_selective8_resume_distill_lr3e4")
config.lr = 3e-4

"""Extend the completed 25/25 phase-1 run from epoch 24 to epoch 27.

This is the control experiment for testing whether three more epochs of the
unchanged phase-1 objective can recover numerical stability by itself.  The
four rank-local model, PartialFC, and SGD momentum states are restored from
the epoch-24 checkpoint.  Only the scheduler horizon is extended, because the
original 24-epoch scheduler is already exhausted.
"""

from easydict import EasyDict as edict

from configs.ms1mv3_r50_no_relu_full_conversion_phase1 import (
    config as _phase1_config,
)


config = edict(_phase1_config.copy())
config.output = (
    "work_dirs/ms1mv3_r50_herpn_full_conversion_phase1_epoch24_plus3")
config.resume = True
config.resume_checkpoint_dir = (
    "work_dirs/ms1mv3_r50_herpn_full_conversion_phase1")

# Preserve the actual epoch-24 optimizer/PartialFC state.  Rebase only the LR
# schedule onto a 27-epoch horizon so epochs 25--27 have non-zero learning
# rates while retaining the same polynomial decay rule.
config.resume_optimizer_state = True
config.resume_rebase_lr_scheduler = True
config.num_epoch = 27

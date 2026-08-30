"""Recover accuracy on the fixed all-polynomial NL9 inference graph.

The source checkpoint has all nine retained PReLUs replaced by degree-2
PReLU-aware HerPN activations.  Each student targets its frozen channel-wise
PReLU on ``[-6, 6]`` and folds to ``A*x^2 + B*x + C`` (one ciphertext square,
or multiplicative depth one, per activation site).  No blend changes during
this run; a fresh optimizer jointly fine-tunes the fixed graph with the
original NL9 embedding as a teacher.
"""

from easydict import EasyDict as edict

from configs.ms1mv3_r50_nl9_prelu_herpn_fast_grouped import (
    config as _source_config,
)


config = edict(_source_config.copy())
config.resume = False
config.output = "work_dirs/ms1mv3_r50_nl9_prelu_herpn_fixed_recovery"
config.backbone_init = (
    "work_dirs/ms1mv3_r50_nl9_prelu_herpn_fast_grouped/model.pt")

config.lr = 3.0e-4
config.warmup_epoch = 1
config.embedding_teacher_network = "r50_nl9"
config.embedding_teacher_checkpoint = "work_dirs/ms1mv3_r50_nl9/model.pt"
config.embedding_distill_weight = 1.0

# Every group is complete before epoch zero.  The scheduled graph therefore
# exactly matches the fully polynomial source checkpoint for the whole run.
config.herpn_group_epochs = (-12.0, -10.0, -8.0, -6.0, -4.0, -2.0)
config.herpn_require_full_conversion = True
config.num_epoch = 10

config.fail_on_nonfinite_val = True
config.max_validation_embedding_abs = 1.0e6


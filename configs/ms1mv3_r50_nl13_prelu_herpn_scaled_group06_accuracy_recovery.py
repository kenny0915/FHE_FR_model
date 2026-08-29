"""Recover six-polynomial IJB-C accuracy from the accepted group-5 prefix.

The prior half-epoch layer3.0 blend produced finite IJB-C embeddings but only
84.72% TAR at FAR=1e-4.  Restore the exact five-group snapshot, condition
layer3.0 for one epoch, blend it over two epochs, then fine-tune the fixed
six-polynomial graph for four epochs while distilling the original NL13
embedding.  Later activations remain PReLU and are not converted in this run.
"""

from easydict import EasyDict as edict

from configs.ms1mv3_r50_nl13_prelu_herpn_scaled_recover_group03 import (
    config as recovery_config,
)


config = edict(recovery_config.copy())
config.resume = False
config.output = (
    "work_dirs/"
    "ms1mv3_r50_nl13_prelu_herpn_scaled_group06_accuracy_recovery")
config.backbone_init = (
    "work_dirs/ms1mv3_r50_nl13_prelu_herpn_scaled_recover_group03/"
    "model_herpn_group_05_bnrecalibrated.pt")
config.backbone_init_herpn_progress = 0.0  # Snapshot blends take precedence.

config.embedding_distill_weight = 1.0
config.embedding_teacher_network = "r50_nl13"
config.embedding_teacher_checkpoint = "work_dirs/ms1mv3_r50_nl13/model.pt"

config.herpn_transition_epochs = 2.0
config.herpn_group_epochs = (
    -10.0,
    -8.0,
    -6.0,
    -4.0,
    -2.0,
    1.0,
    100.0,
    102.0,
    104.0,
    106.0,
    108.0,
    110.0,
    112.0,
)
config.herpn_require_full_conversion = False
config.num_epoch = 7

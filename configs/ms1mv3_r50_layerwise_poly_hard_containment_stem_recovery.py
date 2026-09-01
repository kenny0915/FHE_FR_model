"""Resume the rejected stem boundary for one adaptive recovery attempt.

Epoch-2 checkpoints retain the immutable stem interval and the PReLU graph.
The trainer restores the persisted top-512 rare-tail manifest, conditions for
half an epoch, and reruns the exact complete-domain gate at epoch 2.5.  Only a
passing stem may blend during the remaining half epoch.  Epoch 3 performs the
post-conversion BatchNorm recalibration/prefix audit and saves a resumable hold
checkpoint for the second singleton.
"""

from easydict import EasyDict as edict

from configs.ms1mv3_r50_layerwise_poly_hard_containment_stem import (
    config as stem_config,
)


config = edict(stem_config.copy())
config.resume = True
config.resume_rebase_lr_scheduler = True

config.herpn_transition_epochs = 0.5
config.herpn_group_epochs = (
    2.5,
    *tuple(4.0 + 2.0 * index for index in range(24)),
)
config.layerwise_poly_training_group_limit = 1
config.num_epoch = 4

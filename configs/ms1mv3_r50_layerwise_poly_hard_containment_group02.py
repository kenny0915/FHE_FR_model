"""Resume the accepted stem boundary and prove the second singleton.

This config is launched only after the stem proof has saved epoch-4 distributed
checkpoints in the shared output directory.  On resume, the trainer detects
that ``layer1.0.prelu`` is newly inside the requested conversion frontier,
performs its complete two-orientation provisional scan, and replays the global
top-512 source tails for a half-epoch conditioning probe.  Its immutable
interval must then pass the same complete-domain and polynomial-boundary gates
before a half-epoch blend can begin.  A failed probe stops without weakening
the gate; its recovery config can move the blend boundary later.
"""

from easydict import EasyDict as edict

from configs.ms1mv3_r50_layerwise_poly_hard_containment_stem import (
    config as stem_config,
)


config = edict(stem_config.copy())
config.resume = True
config.resume_rebase_lr_scheduler = True

# Epochs 0--3 belong to the completed stem proof. Resume starts at epoch 4.
# A complete scan is cheap relative to a 5.18M-image epoch, so group 2 first
# receives a 0.5-epoch conditioning probe, then a 0.5-epoch blend only if its
# strict scan passes. Epoch 5 processes BN recalibration/post-audit and holds
# the accepted graph under the persistent tail guard.
config.herpn_transition_epochs = 0.5
config.herpn_group_epochs = (
    2.0,
    4.5,
    *tuple(6.5 + 2.0 * index for index in range(23)),
)
config.layerwise_poly_training_group_limit = 2
config.num_epoch = 6

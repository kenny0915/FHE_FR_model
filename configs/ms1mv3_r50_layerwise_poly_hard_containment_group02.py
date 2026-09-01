"""Resume the accepted stem boundary and prove the second singleton.

This config is launched only after the stem proof has saved epoch-4 distributed
checkpoints in the shared output directory.  On resume, the trainer detects
that ``layer1.0.prelu`` is newly inside the requested conversion frontier,
performs its complete two-orientation provisional scan, and replays the global
top-512 source tails for two full conditioning epochs.  Its immutable interval
must then pass the same complete-domain and polynomial-boundary gates before
the second blend can begin.
"""

from easydict import EasyDict as edict

from configs.ms1mv3_r50_layerwise_poly_hard_containment_stem import (
    config as stem_config,
)


config = edict(stem_config.copy())
config.resume = True
config.resume_rebase_lr_scheduler = True

# Epochs 0--3 belong to the completed stem proof.  Resume starts at epoch 4;
# delaying group 2's blend until epoch 6 preserves two complete local-fit
# epochs (4 and 5) after its provisional full-domain scan.
config.herpn_group_epochs = (
    2.0,
    6.0,
    *tuple(9.0 + 3.0 * index for index in range(23)),
)
config.layerwise_poly_training_group_limit = 2
config.num_epoch = 8

"""Condition the rejected stem tail on the exact inference BatchNorm graph.

The first adaptive recovery replayed the correct source indices but computed
its range loss with train-mode batch moments.  Its complete eval-mode scan
therefore regressed from a 1.002899 to a 1.009815 containment ratio.  Resume
again from the unchanged epoch-2 PReLU checkpoint, keep BatchNorm running
statistics fixed during optimization, and allow affine/conv parameters to
move against the same inference normalization used by the strict gate.
"""

from easydict import EasyDict as edict

from configs.ms1mv3_r50_layerwise_poly_hard_containment_stem_recovery import (
    config as recovery_config,
)


config = edict(recovery_config.copy())
config.freeze_batchnorm_running_stats = True
config.freeze_batchnorm_affine = False

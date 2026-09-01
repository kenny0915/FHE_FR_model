"""Recover the narrowly rejected second singleton without widening its rail.

The half-epoch probe reduced ``layer1.0.prelu`` from a 1.78197 containment
ratio to 1.002056, but the exact gate correctly rejected it.  Resume the same
audited epoch-4 stem boundary, repeat conditioning for one complete epoch,
and use a full epoch for the polynomial blend.  The second group completes at
epoch 6 so BatchNorm recalibration and the prefix audit occur on an integer
epoch boundary; epoch 6 then holds the accepted graph under the tail guard.
"""

from easydict import EasyDict as edict

from configs.ms1mv3_r50_layerwise_poly_hard_containment_group02 import (
    config as group02_config,
)


config = edict(group02_config.copy())
config.herpn_transition_epochs = 1.0
config.herpn_group_epochs = (
    2.0,
    5.0,
    *tuple(7.0 + 2.0 * index for index in range(23)),
)
config.num_epoch = 7

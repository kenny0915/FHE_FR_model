"""Condition the second singleton against an interior rail guard band.

Two exact scans after 0.5 and 1.0 conditioning epochs ended at ratios 1.002056
and 1.002571 with different worst sources.  That migration shows that a hinge
which vanishes exactly at S has no incentive to build durable headroom.  Keep
the immutable acceptance interval unchanged, but start the training-only
penalty at 0.98*S and explicitly prioritize both newly observed rail sources.
"""

from easydict import EasyDict as edict

from configs.ms1mv3_r50_layerwise_poly_hard_containment_group02_recovery import (
    config as recovery_config,
)


config = edict(recovery_config.copy())
config.layerwise_poly_range_guard_ratio = 0.98
config.layerwise_poly_tail_replay_extra_indices = (3005115, 4498665)
# The preceding recovery saved its conditioned PReLU state at epoch 5 before
# the strict gate rejected it.  Resume there, leave one full epoch for the new
# guard-band objective, blend during epoch 6, and hold during epoch 7.
config.herpn_group_epochs = (
    2.0,
    6.0,
    *tuple(8.0 + 2.0 * index for index in range(23)),
)
config.num_epoch = 8

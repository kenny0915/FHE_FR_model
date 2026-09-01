"""Condition the stem with a non-diluted worst-case inference-tail loss.

The eval-BN recovery aligned the training and gate graphs, but a squared
top-16-sample loss still assigned only about 2e-5 penalty to the single rail
pixel and the exact maximum regressed to a 1.018422 ratio.  This run keeps the
same immutable interval and epoch-2 PReLU checkpoint, uses a linear per-batch
L-infinity hinge, and oversamples the eight most severe manifest sources.
"""

from easydict import EasyDict as edict

from configs.ms1mv3_r50_layerwise_poly_hard_containment_stem_evalbn import (
    config as evalbn_config,
)


config = edict(evalbn_config.copy())
config.layerwise_poly_range_penalty_mode = "containment_max"
config.layerwise_poly_tail_replay_priority_count = 8
config.layerwise_poly_tail_replay_priority_repeats = 64

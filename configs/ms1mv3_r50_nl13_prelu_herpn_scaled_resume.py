"""Resume causal NL13 conversion after a guarded tail-ratio stop.

The first three converted activations remained numerically bounded.  The
fourth activation had a single observed tail 6.72 times the robust 99.95th
percentile interval, so permit tails up to 8 while retaining calibration to
the full observed maximum and the existing absolute scale/growth guards.
"""

from easydict import EasyDict as edict

from configs.ms1mv3_r50_nl13_prelu_herpn_scaled import config as base_config


config = edict(base_config.copy())
config.resume = True
config.layerwise_poly_max_tail_ratio = 8.0

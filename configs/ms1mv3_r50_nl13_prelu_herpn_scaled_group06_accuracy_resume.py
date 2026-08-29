"""Resume group-6 accuracy recovery with its conditioned tail interval.

One local-fit epoch reduced layer3.0's robust magnitude from 133.55 to 2.98
and its observed maximum from 1842.48 to 215.94.  Flooring the public interval
to keep that maximum within the tail-ratio limit requires a 4.979x expansion,
so permit 5.1x for this recovery while retaining every other safety guard.
"""

from easydict import EasyDict as edict

from configs.ms1mv3_r50_nl13_prelu_herpn_scaled_group06_accuracy_recovery import (
    config as recovery_config,
)


config = edict(recovery_config.copy())
config.resume = True
config.layerwise_poly_max_tail_scale_expansion = 5.1

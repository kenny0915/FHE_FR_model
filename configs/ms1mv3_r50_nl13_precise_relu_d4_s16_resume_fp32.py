"""Resume the degree-4 S=16 curriculum in FP32.

The original FP16 run reached the completed degree-16 stage, but a rare
out-of-interval training batch made the degree-16 polynomial exceed FP16's
finite range when its FP32 result was cast back to the autocast dtype.  Keep
the exact, unclipped polynomial forward and resume from the epoch-4 sharded
checkpoint in FP32 instead.
"""

from copy import deepcopy

from configs.ms1mv3_r50_nl13_precise_relu_d4_s16 import config as d4_config


config = deepcopy(d4_config)
config.resume = True
config.fp16 = False

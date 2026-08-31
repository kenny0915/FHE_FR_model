"""NL13 degree-4 activation range ablation on the interval [-32, 32].

This run shares the degree-4 curriculum and teacher with the scale-16 model.
It tests whether wider tail coverage is worth the larger central approximation
error while retaining nonlinear multiplicative depth two.
"""

from copy import deepcopy

from configs.ms1mv3_r50_nl13_precise_relu_d4_s16 import config as scale16_config


config = deepcopy(scale16_config)
config.output = "work_dirs/ms1mv3_r50_nl13_precise_relu_d4_s32"
config.precise_relu_input_scale = 32.0

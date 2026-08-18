"""Scale-24 Alpha10-to-Alpha7 R50 comparison experiment."""

from copy import deepcopy

from configs.ms1mv3_r50_precise_relu_alpha7_s16 import config as scale16_config


config = deepcopy(scale16_config)
config.output = "work_dirs/ms1mv3_r50_precise_relu_alpha7_s24"
config.precise_relu_input_scale = 24.0
config.precise_relu_approximation_error_bound = 0.1875

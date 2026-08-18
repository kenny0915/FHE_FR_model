"""Scale-16 variant of the progressive PreciseReLU R50 experiment."""

from copy import deepcopy

from configs.ms1mv3_r50_precise_relu_s8 import config as scale8_config


config = deepcopy(scale8_config)
config.output = "work_dirs/ms1mv3_r50_precise_relu_s16_d4"
config.precise_relu_input_scale = 16.0

"""Resume the degree-8 S=16 curriculum in FP32 after FP16 overflow."""

from copy import deepcopy

from configs.ms1mv3_r50_nl13_precise_relu_d8_s16 import config as d8_config


config = deepcopy(d8_config)
config.resume = True
config.fp16 = False

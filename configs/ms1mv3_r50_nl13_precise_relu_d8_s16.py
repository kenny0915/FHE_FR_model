"""NL13 PReLU teacher distilled to degree-8 activations on [-16, 16].

The final zero-preserving degree-8 ChebyReLU has nonlinear multiplicative
depth three.  This run isolates whether its lower in-range approximation error
recovers IJB-C low-FAR geometry relative to degree 4 at the same public range.
"""

from copy import deepcopy

from configs.ms1mv3_r50_nl13_precise_relu_d4_s16 import config as d4_config


config = deepcopy(d4_config)
config.output = "work_dirs/ms1mv3_r50_nl13_precise_relu_d8_s16"
config.precise_relu_lower_degrees = (16, 8)
config.precise_relu_stage_epochs = (2, 6)

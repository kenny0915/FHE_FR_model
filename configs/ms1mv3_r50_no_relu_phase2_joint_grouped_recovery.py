"""Joint epoch-23 recovery with independent Conv/HerPN optimization.

All 53 convolution tensors and all coefficients in the 25 degree-2 HerPNs are
updated in the same end-to-end backward pass.  Their learning rates and FP64
gradient-norm clips are independent, so a catastrophic tail in one group
cannot round the other group's update down to zero.  BN remains immutable.

The approximation target is PReLU on [-6, 6].  Training uses the same
straight-through numerical surrogate as the first joint run; evaluation and
FHE inference remain the exact unclipped Ax^2 + Bx + C graph with unchanged
multiplicative depth.
"""

from easydict import EasyDict as edict

from configs.ms1mv3_r50_no_relu_phase2_joint_recovery import (
    config as _joint_config,
)


config = edict(_joint_config.copy())
config.output = (
    "work_dirs/"
    "ms1mv3_r50_herpn_full_conversion_phase2_joint_grouped_recovery")

# Conv and HerPN still update together.  Only their optimizer scale and norm
# reduction are separated.  H0/H1/H2 begin at one; H2 is not attenuated.
config.split_conv_herpn_optimizer = True
config.herpn_lr_multiplier = 10.0
config.separate_conv_herpn_gradient_clip = True
config.conv_gradient_clip = 1.0
config.herpn_gradient_clip = 0.1
config.stable_gradient_clip = True
config.gradient_clip_type = "norm"

# Absolute scheduled rates start at 1e-6 for Conv and 1e-5 for HerPN.  Their
# maximum per-step L2 update is therefore balanced at 1e-6 before weight decay.
config.lr = 1e-6
config.momentum = 0.0
config.num_epoch = 2

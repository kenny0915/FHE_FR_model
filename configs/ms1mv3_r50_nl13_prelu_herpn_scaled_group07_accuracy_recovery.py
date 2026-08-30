"""Attempt a slow, distilled seventh polynomial from the accepted group 6.

Resume the final epoch-7 distributed state whose inference graph has exactly
six polynomial activations. Condition ``layer3.3.prelu`` for one epoch, blend
it over two epochs, then fine-tune the fixed-seven graph for four epochs with
baseline NL13 embedding distillation. Later activations remain PReLU.
"""

from easydict import EasyDict as edict

from configs.ms1mv3_r50_nl13_prelu_herpn_scaled_group06_accuracy_finetune import (
    config as group06_config,
)


config = edict(group06_config.copy())
config.resume = True
config.layerwise_poly_conditioning_range_loss_weight = 1.0
config.herpn_distill_loss_weight = 1.0
config.herpn_group_epochs = (
    -10.0,
    -8.0,
    -6.0,
    -4.0,
    -2.0,
    1.0,
    8.0,
    100.0,
    102.0,
    104.0,
    106.0,
    108.0,
    110.0,
)
config.num_epoch = 14

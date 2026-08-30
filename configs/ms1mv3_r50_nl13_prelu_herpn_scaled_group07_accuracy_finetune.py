"""Fine-tune the fixed-seven polynomial graph after its slow transition.

Resume epoch 10 from ``group07_accuracy_recovery``.  At that point the first
seven activations are fully polynomial.  Keep the eighth transition far in
the future so the remaining six activations stay PReLU, disable the local
range objective, and use only baseline-NL13 embedding distillation while the
fixed graph adapts.
"""

from easydict import EasyDict as edict

from configs.ms1mv3_r50_nl13_prelu_herpn_scaled_group07_accuracy_recovery import (
    config as group07_config,
)


config = edict(group07_config.copy())
config.resume = True
config.layerwise_poly_conditioning_range_loss_weight = 1.0e-8
config.herpn_distill_loss_weight = 0.0
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

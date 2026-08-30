"""Conservative recovery with the known-finite NL9 BatchNorm state fixed.

All nine PReLU sites remain degree-2 HerPN polynomials targeting the frozen
channel-wise PReLU on ``[-6, 6]``.  The source graph was finite on full IJB-C;
freezing both BatchNorm running statistics and affine parameters prevents
fine-tuning from invalidating that safe normalization state.  Only public
weights and the fixed polynomial coefficients adapt, so encrypted inference
still uses nine quadratics at multiplicative depth one per site.
"""

from easydict import EasyDict as edict

from configs.ms1mv3_r50_nl9_prelu_herpn_fixed_recovery import (
    config as _recovery_config,
)


config = edict(_recovery_config.copy())
config.output = (
    "work_dirs/ms1mv3_r50_nl9_prelu_herpn_fixed_recovery_bnfreeze")
config.lr = 5.0e-5
config.embedding_distill_weight = 5.0
config.freeze_batchnorm_running_stats = True
config.freeze_batchnorm_affine = True


"""Resume the fully blended group-6 graph for accuracy fine-tuning.

The epoch-3 state is immediately before group-6 completion bookkeeping.  Keep
group 7 at blend zero, use the staged-training optimizer topology for exact
resume compatibility, but devote the remaining updates to baseline embedding
distillation at 10% backbone LR.  The numerically positive group-7 range weight
only satisfies the provisional-schedule invariant and is intentionally
negligible because group 7 is outside this experiment's inference graph.
"""

from easydict import EasyDict as edict

from configs.ms1mv3_r50_nl13_prelu_herpn_scaled_group06_accuracy_resume import (
    config as resume_config,
)


config = edict(resume_config.copy())
config.layerwise_poly_conditioning_backbone_lr_scale = 0.1
config.layerwise_poly_conditioning_range_loss_weight = 1e-8
config.herpn_distill_loss_weight = 0.0

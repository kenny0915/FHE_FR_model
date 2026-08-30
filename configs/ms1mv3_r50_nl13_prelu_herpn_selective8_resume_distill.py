"""Resume the tested selective-eight model without replacing its class head.

Unlike a backbone-only recovery, this run restores the epoch-8 PartialFC and
optimizer states that produced the finite 89.50% IJB-C checkpoint.  The LR
scheduler is rebased over four extra epochs and normalized embeddings are
distilled from the accepted selective-seven teacher.  The eight active sites
remain folded degree-2 polynomials throughout; later sites remain PReLU.
"""

from easydict import EasyDict as edict

from configs.ms1mv3_r50_nl13_prelu_herpn_selective8_layer42_layer41_recovered import (
    config as _recovered_config,
)


config = edict(_recovered_config.copy())
config.output = (
    "work_dirs/"
    "ms1mv3_r50_nl13_prelu_herpn_selective8_resume_distill")
config.resume = True
config.resume_rebase_lr_scheduler = True
config.num_epoch = 12

config.embedding_distill_weight = 5.0
config.embedding_teacher_network = "r50_nl13_prelu_herpn"
config.embedding_teacher_checkpoint = (
    "work_dirs/ms1mv3_r50_nl13_prelu_herpn_selective7_layer42/"
    "model_epoch_07.pt")
config.layerwise_poly_training_group_limit = 8
config.layerwise_poly_conditioning_range_loss_weight = 1e-8
config.herpn_distill_loss_weight = 0.0

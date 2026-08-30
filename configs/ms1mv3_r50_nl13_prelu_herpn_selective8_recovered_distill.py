"""Recover the fully converted selective-eight model with embedding distillation.

All eight selected activations start and remain at a 100% polynomial blend.
The frozen selective-seven checkpoint is used only as a plaintext training
teacher for normalized embedding direction; it is absent from inference.
Every retained activation is still one folded degree-2 polynomial on its
previously calibrated public ``[-S, S]`` interval.
"""

from easydict import EasyDict as edict

from configs.ms1mv3_r50_nl13_prelu_herpn_selective8_layer42_layer41_recovered import (
    config as _recovered_config,
)


config = edict(_recovered_config.copy())
config.output = (
    "work_dirs/"
    "ms1mv3_r50_nl13_prelu_herpn_selective8_recovered_distill")
config.backbone_init = (
    "work_dirs/"
    "ms1mv3_r50_nl13_prelu_herpn_selective8_layer42_layer41_recovered/"
    "model.pt")

# The first eight groups are already fully converted in backbone_init.  Their
# negative starts make that state explicit and avoid a second calibration or
# blend cycle.  The five remaining PReLUs stay outside this training run.
config.herpn_group_epochs = (
    -16.0, -14.0, -12.0, -10.0, -8.0, -6.0, -4.0, -2.0,
    100.0, 102.0, 104.0, 106.0, 108.0,
)
config.layerwise_poly_training_group_limit = 8

# Low-LR angular recovery.  The unit-embedding teacher loss directly targets
# the geometry used by cosine/template matching and adds nothing to inference.
config.lr = 1e-4
config.warmup_epoch = 0
config.num_epoch = 6
config.embedding_distill_weight = 5.0
config.embedding_teacher_network = "r50_nl13_prelu_herpn"
config.embedding_teacher_checkpoint = (
    "work_dirs/ms1mv3_r50_nl13_prelu_herpn_selective7_layer42/"
    "model_epoch_07.pt")

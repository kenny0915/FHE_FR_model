"""NL13 PReLU teacher distilled to degree-4 activations on [-16, 16].

All thirteen retained PReLUs keep their learned channel-wise linear slope and
replace only the ReLU component.  The final zero-preserving degree-4
ChebyReLU has nonlinear multiplicative depth two.  Alpha10 -> degree 16 ->
degree 8 -> degree 4 provides a stable continuation from the trained NL13
checkpoint; all polynomial evaluations use the public interval [-16, 16].
"""

from copy import deepcopy

from configs.ms1mv3_r50_precise_relu_s16 import config as r50_scale16_config


config = deepcopy(r50_scale16_config)
config.arch_config = "nl13"
config.output = "work_dirs/ms1mv3_r50_nl13_precise_relu_d4_s16"
config.backbone_init = "work_dirs/ms1mv3_r50_nl13/model.pt"

# Keep the polynomial embedding geometry close to the 96.15% strict-IJB-C
# NL13 PReLU control while the fresh PartialFC classifier adapts.
config.embedding_teacher_network = "r50_nl13"
config.embedding_teacher_checkpoint = config.backbone_init
config.embedding_distill_weight = 1.0

# Nano4 uses eight H200s.  This preserves the base recipe's linear LR scaling
# at global batch 1024 and assigns one data-loader worker to each requested CPU
# core across the eight ranks.
config.batch_size = 128
config.lr = 0.01
config.num_workers = 8
config.validation_batch_size = 256
config.fail_on_nonfinite_val = True

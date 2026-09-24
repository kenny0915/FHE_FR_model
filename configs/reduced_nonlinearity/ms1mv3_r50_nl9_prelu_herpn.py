"""Convert the temporary MS1MV3 NL9 checkpoint to nine degree-2 polynomials.

Approximation target for each frozen channel-wise PReLU slope ``a``:

    PReLU_a(x) = a*x + (1-a)*ReLU(x),  x in [-6, 6].

The student replaces the ReLU component with basis-normalized HerPN.  The
central empirical mass (normally about [-3, 3] after normalization) drives the
fit, while [-6, 6] is monitored and penalized as the safety interval.  At FHE
inference every activation folds to ``A*x^2 + B*x + C`` and needs one square.

This experiment intentionally initializes from the current temporary NL9
``model.pt``.  Copy that checkpoint to an immutable path before launching if
the PReLU training job is still writing it.
"""

from easydict import EasyDict as edict


config = edict()
config.margin_list = (1.0, 0.5, 0.0)
config.network = "r50_nl9_prelu_herpn"
config.arch_config = "nl9"
config.resume = False
config.output = "work_dirs/ms1mv3_r50_nl9_prelu_herpn"
config.embedding_size = 512
config.sample_rate = 1.0

# Quadratic branches and their range losses are evaluated in FP32 for the
# first stability run.  AMP can be tested only after complete conversion.
config.fp16 = False
config.batch_size = 128
config.lr = 0.001
config.momentum = 0.9
config.weight_decay = 5e-4
config.selective_weight_decay = True
config.gradient_clip = 1.0
config.gradient_clip_scope = "backbone"
config.warmup_epoch = 1
config.dali = False
config.verbose = 2000

# Blend zero is exactly this PReLU checkpoint.  The same frozen NL9 model is
# used as an embedding teacher throughout conversion.
config.backbone_init = "work_dirs/ms1mv3_r50_nl9/model.pt"
config.embedding_teacher_network = "r50_nl9"
config.embedding_teacher_checkpoint = config.backbone_init
config.embedding_distill_weight = 1.0

config.herpn_initial_progress = 0.0
config.herpn_bn_eps = 1e-4
config.herpn_range_limit = 6.0
config.herpn_range_loss_weight = 0.05
config.prelu_herpn_distill_eps = 1e-4
config.herpn_distill_loss_weight = 1.0

# Convert one activation at a time in strict forward order.  The first two
# epochs fit all zero-blend students locally.  Each two-epoch conversion is
# followed by one full epoch before the next transition, and every completed
# singleton receives globally synchronized BatchNorm recalibration.
config.herpn_stage_epochs = ()
config.herpn_conversion_groups = (
    ("prelu",),
    ("layer1.0.prelu",),
    ("layer2.0.prelu",),
    ("layer3.0.prelu",),
    ("layer3.3.prelu",),
    ("layer3.9.prelu",),
    ("layer3.13.prelu",),
    ("layer4.0.prelu",),
    ("layer4.2.prelu",),
)
config.herpn_group_epochs = (2, 5, 8, 11, 14, 17, 20, 23, 26)
config.herpn_transition_epochs = 2.0
config.herpn_bn_recalibration_batches = 1000
config.herpn_save_after_group = True
config.herpn_require_full_conversion = True

config.sync_bn = True
config.broadcast_buffers = True
config.ddp_fp16_compress = False
config.check_finite_grads = True
config.fail_on_nonfinite_val = True
config.max_validation_embedding_abs = 1e6
config.save_all_states = True
config.checkpoint_interval_epochs = 1
config.save_epoch_models = True
config.epoch_model_interval = 1

config.rec = "./ms1m-retinaface-t1"
config.num_classes = 93431
config.num_image = 5179510
# The final transition completes at epoch 28; retain six fully polynomial
# epochs for joint ArcFace, embedding-teacher, and local-teacher fine-tuning.
config.num_epoch = 34
config.val_targets = ["lfw", "cfp_fp", "agedb_30"]

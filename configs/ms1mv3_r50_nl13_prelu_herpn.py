"""Fast grouped conversion of temporary MS1MV3 NL13 PReLUs to HerPN.

For each frozen channel-wise slope ``a``, the degree-2 student targets

    PReLU_a(x) = a*x + (1-a)*ReLU(x),  x in [-6, 6].

The empirical central region near [-3, 3] drives local distillation, while
[-6, 6] is the monitored and penalized safety interval.  Basis-normalized
HerPN folds to ``A*x^2 + B*x + C`` for FHE inference.

To reduce runtime relative to the singleton NL9 experiment, shallow
activations are converted together, Layer3 uses pairs, and only sensitive
Layer4 activations remain singleton.  A second full embedding-teacher backbone
is deliberately disabled; every wrapper still contains its frozen local PReLU
teacher and keeps relative activation distillation active after conversion.
"""

from easydict import EasyDict as edict


config = edict()
config.margin_list = (1.0, 0.5, 0.0)
config.network = "r50_nl13_prelu_herpn"
config.arch_config = "nl13"
config.resume = False
config.output = "work_dirs/ms1mv3_r50_nl13_prelu_herpn"
config.embedding_size = 512
config.sample_rate = 1.0

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

# Use the current temporary NL13 result.  Copy it to an immutable filename and
# update this path first if another process is still writing model.pt.
config.backbone_init = "work_dirs/ms1mv3_r50_nl13/model.pt"
config.embedding_distill_weight = 0.0

config.herpn_initial_progress = 0.0
config.herpn_bn_eps = 1e-4
config.herpn_range_limit = 6.0
config.herpn_range_loss_weight = 0.05
config.prelu_herpn_distill_eps = 1e-4
config.herpn_distill_loss_weight = 1.0

config.herpn_stage_epochs = ()
config.herpn_conversion_groups = (
    # Expensive high-resolution activations share one transition.
    ("prelu", "layer1.0.prelu", "layer1.2.prelu"),
    ("layer2.0.prelu", "layer2.3.prelu"),
    # Preserve forward order through the long stage, two at a time.
    ("layer3.0.prelu", "layer3.3.prelu"),
    ("layer3.6.prelu", "layer3.9.prelu"),
    ("layer3.13.prelu",),
    # Keep the final stage conservative because it is closest to embeddings.
    ("layer4.0.prelu",),
    ("layer4.1.prelu",),
    ("layer4.2.prelu",),
)
# Epoch 0 locally fits every zero-blend student.  Each group then gets one
# blend epoch and one recovery epoch before the next group starts.
config.herpn_group_epochs = (1, 3, 5, 7, 9, 11, 13, 15)
config.herpn_transition_epochs = 1.0
config.herpn_bn_recalibration_batches = 500
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
# Conversion completes at epoch 16, followed by four fully polynomial epochs.
config.num_epoch = 20
config.val_targets = ["lfw", "cfp_fp", "agedb_30"]

from easydict import EasyDict as edict


config = edict()
config.margin_list = (1.0, 0.5, 0.0)
config.network = "r50_quadratic"
config.resume = False
config.output = "work_dirs/ms1mv3_r50_quadratic"
config.embedding_size = 512
config.sample_rate = 1.0
config.fp16 = False
config.momentum = 0.9
config.weight_decay = 5e-4
config.gradient_clip = 0.5
config.batch_size = 128
config.lr = 0.005
config.verbose = 2000
config.dali = False

# Initialize from the original trained PReLU R50.  The polynomial target is
# the channel-wise pretrained PReLU over x in [-6, 6]:
#   z = x / 6
#   y = 6 * (a*z^2 + b*z + c)
# This remains a degree-2 encrypted activation with one sequential
# ciphertext-ciphertext multiplication for z^2.
config.backbone_init = "work_dirs/ms1mv3_r50/model.pt"
config.quadratic_input_scale = 6.0
config.quadratic_range_limit = 6.0
# The preceding IResNet BN makes activation inputs approximately N(0, 1).
# Initialize the PReLU's absolute-value component with the Gaussian-weighted
# Hermite fit |x| ~= k*(x^2 + 1), k=1/sqrt(2*pi), while monitoring the safety
# interval [-6, 6]. The normalized parameters are folded back to A*x^2+B*x+C.
config.quadratic_abs_init = 0.3989422804014327
config.herpn_initial_progress = 0.0

# train_v2 currently uses the herpn_* schedule protocol for every progressive
# polynomial backbone.  Later high-sensitivity activations are converted in
# pairs, then layer4 is converted one activation at a time.
config.herpn_stage_epochs = ()
config.herpn_conversion_groups = (
    ("prelu",),
    ("layer1.0.prelu", "layer1.1.prelu", "layer1.2.prelu"),
    ("layer2.0.prelu", "layer2.1.prelu"),
    ("layer2.2.prelu", "layer2.3.prelu"),
    ("layer3.0.prelu", "layer3.1.prelu"),
    ("layer3.2.prelu", "layer3.3.prelu"),
    ("layer3.4.prelu", "layer3.5.prelu"),
    ("layer3.6.prelu", "layer3.7.prelu"),
    ("layer3.8.prelu", "layer3.9.prelu"),
    ("layer3.10.prelu", "layer3.11.prelu"),
    ("layer3.12.prelu", "layer3.13.prelu"),
    ("layer4.0.prelu",),
    ("layer4.1.prelu",),
    ("layer4.2.prelu",),
)
config.herpn_group_epochs = tuple(range(1, 29, 2))
config.herpn_transition_epochs = 1.0
config.herpn_bn_recalibration_batches = 200
config.herpn_require_full_conversion = True
config.herpn_range_loss_weight = 0.1
# ProgressiveQuadraticActivation keeps this PReLU-teacher loss active even
# after an activation reaches blend=1.
config.herpn_distill_loss_weight = 0.5

config.sync_bn = True
config.broadcast_buffers = True
config.check_finite_grads = True
# Avoid converting large, finite FP32 gradients to Inf during DDP reduction.
config.ddp_fp16_compress = False
config.save_all_states = True
config.checkpoint_interval_epochs = 1
config.save_epoch_models = True
config.epoch_model_interval = 1

config.rec = "./ms1m-retinaface-t1"
config.num_classes = 93431
config.num_image = 5179510
# The last conversion completes at epoch 28; retain four full-polynomial
# fine-tuning epochs.
config.num_epoch = 32
config.warmup_epoch = 2
config.val_targets = ["lfw", "cfp_fp", "agedb_30"]

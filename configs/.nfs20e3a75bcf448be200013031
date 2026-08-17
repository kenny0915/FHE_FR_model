"""Fine-tune fully gated PoolFormer-S24 from LayerNorm to fixed affine maps.

The accepted LayerNorm checkpoint is loaded exactly.  Before optimization,
200 representative batches fit ``a_c*x_c+b_c`` to every one of the 49
LayerNorm outputs by per-channel least squares.  The LayerNorm teacher then
decays over eight epochs, followed by four epochs using only the fixed affine
student.  At inference these affine maps contain only plaintext per-channel
scales and biases and can be folded into adjacent linear operations where the
graph layout permits.

The 24 SimpleGates are unchanged and remain exact degree-2 products.
"""

from easydict import EasyDict as edict


config = edict()
config.margin_list = (1.0, 0.0, 0.4)
config.network = "poolformer_fully_gated_affine_s24"
config.resume = True
config.output = "work_dirs/ms1mv3_poolformer_s24_fully_gated_affine_fp32"
config.embedding_size = 512
config.sample_rate = 1.0

# Strict warm start from the accepted fully gated LayerNorm backbone.
config.backbone_init = (
    "work_dirs/ms1mv3_poolformer_s24_fully_gated_fp32/model.pt")

# Fit each affine student to its LayerNorm teacher before the transition.
# With four GPUs and batch_size=128, 200 batches cover about 102k faces.
config.affine_calibration_batches = 200
config.affine_calibration_ridge = 1e-6

# Generic progressive-normalization hooks drive LN weight from one to zero.
config.prepbn_decay_epochs = 8
config.prepbn_require_full_transition = True
config.validate_after_prepbn_transition = True
config.prepbn_bn_stat_epochs = 0
config.final_verification_after_prepbn = True

config.fp16 = False
config.gradient_clip = 0.01
config.gradient_clip_type = "norm"
config.gradient_clip_scope = "backbone"
config.check_finite_grads = True
config.fail_on_nonfinite_val = True

config.optimizer = "adamw"
config.lr = 1e-4
config.weight_decay = 0.001
config.momentum = 0.9
config.batch_size = 128
config.warmup_epoch = 0
config.num_epoch = 12

# The head still contains BatchNorm, so preserve the established synchronized
# training behavior.  The 49 converted backbone normalization sites do not.
config.sync_bn = True
config.broadcast_buffers = True
config.ddp_fp16_compress = False
config.save_all_states = True
config.checkpoint_interval_epochs = 1
config.save_epoch_models = True
config.epoch_model_interval = 1

config.verbose = 2000
config.frequent = 10
config.dali = False
config.dali_aug = False
config.gradient_acc = 1

config.rec = "./ms1m-retinaface-t1"
config.num_classes = 93431
config.num_image = 5179510
config.interclass_filtering_threshold = 0
config.val_targets = ["lfw", "cfp_fp", "agedb_30"]

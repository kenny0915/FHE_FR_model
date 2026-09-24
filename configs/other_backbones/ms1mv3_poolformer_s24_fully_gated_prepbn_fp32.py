"""Convert the accepted fully gated PoolFormer-S24 from LayerNorm to RepBN.

The starting graph is exactly the trained channel-wise LayerNorm backbone.
Over eight epochs, PRepBN linearly changes each of its 49 normalization sites
from the exact LayerNorm teacher to ``BatchNorm2d(x) + eta*x``.  The remaining
four optimization epochs use only RepBN, followed by a representative-data BN
statistics refresh.  Frozen RepBN is a channel-wise affine map and introduces
no encrypted-data-dependent division or square root.

SimpleGate is already the final degree-2 activation, so this experiment does
not blend or otherwise alter any activation.  Its approximation target is not
changed: each gate computes the exact product of its two channel halves.  The
24 sequential gates still determine the encrypted multiplicative-depth budget.
"""

from easydict import EasyDict as edict


config = edict()
config.margin_list = (1.0, 0.0, 0.4)
config.network = "poolformer_fully_gated_prepbn_s24"
config.resume = False
config.output = "work_dirs/ms1mv3_poolformer_s24_fully_gated_prepbn_fp32"
config.embedding_size = 512
config.sample_rate = 1.0

# Strictly map every parameter from the accepted LayerNorm checkpoint.  Only
# the new BN running state and learned RepBN eta scalars are initialized anew.
config.backbone_init = (
    "work_dirs/ms1mv3_poolformer_s24_fully_gated_fp32/model.pt")

# Match the stable all-SimpleGate run.  PRepBN is a normalization-only change.
config.fp16 = False
config.gradient_clip = 0.01
config.gradient_clip_type = "norm"
config.gradient_clip_scope = "backbone"
config.check_finite_grads = True
config.fail_on_nonfinite_val = True

# RepBN(x) = BN(x) + eta*x with one eta scalar per norm.  The official model
# initializes eta to one, but that makes the untrained student path non-finite
# in this 24-gate quadratic backbone.  Zero starts from normalized BN and lets
# eta learn the residual contribution during the eight-epoch transition.
config.repbn_bn_eps = 1e-5
config.repbn_bn_momentum = 0.1
config.repbn_eta_init = 0.0
config.prepbn_decay_epochs = 8
config.prepbn_require_full_transition = True
config.validate_after_prepbn_transition = True
config.prepbn_bn_stat_epochs = 1
config.final_verification_after_prepbn = True

# Conservative warm-start fine-tuning.  Four pure-RepBN epochs follow the
# transition before the final running-statistics refresh.
config.optimizer = "adamw"
config.lr = 1e-4
config.weight_decay = 0.001
config.momentum = 0.9
config.batch_size = 128
config.warmup_epoch = 0
config.num_epoch = 12

# Synchronize training-time BN statistics across the four V100 workers and
# keep checkpoints at every epoch so accuracy can be audited across gamma.
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

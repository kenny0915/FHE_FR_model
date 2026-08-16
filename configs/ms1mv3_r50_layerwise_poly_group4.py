"""MS1Mv3 R50 conversion with up to four PReLUs per stable group.

Each group is calibrated in one representative-data pass. During local fit,
only the current group's polynomial coefficients receive backbone gradients.
The group then blends for one epoch with a 0.1x backbone learning rate, after
which all BatchNorm modules downstream of the group's earliest activation are
recalibrated. Seven final epochs jointly fine-tune the polynomial network.

The approximation target at activation i is its frozen channel-wise PReLU on
[-S_i, S_i]. Degree 2 needs one sequential ciphertext square after folding.
All calibration, distillation, blending, and freezing logic is plaintext-only
training machinery and is absent from the folded FHE inference graph.
"""

from easydict import EasyDict as edict


config = edict()
config.margin_list = (1.0, 0.5, 0.0)
config.network = "r50_layerwise_poly"
config.resume = False
config.output = "work_dirs/ms1mv3_r50_layerwise_poly_group4_d2"
config.embedding_size = 512
config.sample_rate = 1.0
config.fp16 = False
config.momentum = 0.9
config.weight_decay = 5e-4
config.selective_weight_decay = True
config.gradient_clip = 0.5
config.batch_size = 128
config.lr = 0.005
config.verbose = 2000
config.dali = False

# Blend zero reproduces the pretrained PReLU teacher exactly.
config.backbone_init = "work_dirs/ms1mv3_r50/model.pt"
config.herpn_initial_progress = 0.0
config.layerwise_poly_degree = 2
config.layerwise_poly_initial_scale = 1.0
config.layerwise_poly_distill_eps = 1e-4

# One full distributed pass measures every activation in the pending group.
# The larger 1.5 margin is retained from the earlier interval correction; the
# tail guard remains active instead of trying to absorb genuine runaways.
config.layerwise_poly_range_calibration_batches = 0
config.layerwise_poly_range_margin = 1.5
config.layerwise_poly_min_scale = 1e-3
config.layerwise_poly_range_quantile = 0.999
config.layerwise_poly_quantile_samples = 65536
config.layerwise_poly_range_holdout_fraction = 0.05
config.layerwise_poly_max_tail_ratio = 8.0
config.layerwise_poly_max_scale_growth = 16.0
config.layerwise_poly_max_input_scale = 1e5

config.herpn_range_loss_weight = 0.1
config.herpn_distill_loss_weight = 1.0
config.herpn_bn_recalibration_batches = 1000
config.herpn_save_after_group = True
config.herpn_require_full_conversion = True
config.herpn_stage_epochs = ()

# Freeze convolution/normalization affine gradients during local fit. The
# current polynomial group and classifier remain trainable. During blending
# and final joint fine-tuning, scale only the non-polynomial backbone LR.
config.layerwise_poly_staged_training = True
config.layerwise_poly_freeze_backbone_during_local_fit = True
config.layerwise_poly_blend_backbone_lr_scale = 0.1
config.layerwise_poly_final_backbone_lr_scale = 0.1

# Preserve forward order and stage boundaries. R50 has 25 PReLUs, producing
# five four-activation groups followed by groups of two and three.
config.herpn_conversion_groups = (
    (
        "prelu",
        "layer1.0.prelu",
        "layer1.1.prelu",
        "layer1.2.prelu",
    ),
    (
        "layer2.0.prelu",
        "layer2.1.prelu",
        "layer2.2.prelu",
        "layer2.3.prelu",
    ),
    (
        "layer3.0.prelu",
        "layer3.1.prelu",
        "layer3.2.prelu",
        "layer3.3.prelu",
    ),
    (
        "layer3.4.prelu",
        "layer3.5.prelu",
        "layer3.6.prelu",
        "layer3.7.prelu",
    ),
    (
        "layer3.8.prelu",
        "layer3.9.prelu",
        "layer3.10.prelu",
        "layer3.11.prelu",
    ),
    (
        "layer3.12.prelu",
        "layer3.13.prelu",
    ),
    (
        "layer4.0.prelu",
        "layer4.1.prelu",
        "layer4.2.prelu",
    ),
)

# Group 1 receives two local-fit epochs. Later groups receive one local-fit
# epoch after the previous group completes, followed by one blend epoch.
config.herpn_group_epochs = (2, 4, 6, 8, 10, 12, 14)
config.herpn_transition_epochs = 1.0

config.sync_bn = True
config.broadcast_buffers = True
config.check_finite_grads = True
config.ddp_fp16_compress = False
config.fail_on_nonfinite_val = True
config.save_all_states = True
config.checkpoint_interval_epochs = 1
config.save_epoch_models = True
config.epoch_model_interval = 1

config.rec = "./ms1m-retinaface-t1"
config.num_classes = 93431
config.num_image = 5179510
# Last group completes at epoch 15; retain seven fully polynomial epochs.
config.num_epoch = 22
config.warmup_epoch = 1
config.val_targets = ["lfw", "cfp_fp", "agedb_30"]

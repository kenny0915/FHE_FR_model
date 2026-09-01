"""Finish the legacy R50 conversion while quarantining non-finite batches.

This is a feasibility phase, not an accuracy or numerical-safety acceptance
run.  It resumes the last finite four-rank distributed checkpoint from the
legacy ``ms1mv3_r50_herpn`` experiment, completes the original 25-site HerPN
schedule, and writes all new states to a separate work directory.

The inference graph is unchanged: every completed activation remains the
original channel-wise degree-2 HerPN polynomial targeting PReLU on the
monitored interval ``[-6, 6]``.  No clamp or data-dependent inference branch
is introduced.  During training only, a batch with non-finite embeddings,
loss, or gradients is synchronously skipped on every rank; BatchNorm running
buffers are restored so the rejected forward cannot poison later inference.
"""

from easydict import EasyDict as edict

from configs.ms1mv3_r50_no_relu import config as _base_config


config = edict(_base_config.copy())

# Read the finite epoch-15 resume state (saved after epoch 14) without
# overwriting the original experiment.  The rank count must remain four
# because PartialFC and optimizer state are rank-local.
config.resume = True
config.resume_checkpoint_dir = "work_dirs/ms1mv3_r50_herpn"
config.output = "work_dirs/ms1mv3_r50_herpn_full_conversion_phase1"
config.tensorboard = False
# The legacy checkpoint predates the current selective optimizer grouping, so
# its momentum parameter groups cannot be loaded safely by position.  Restore
# all model/PartialFC tensors, start clean optimizer buffers, and place the
# scheduler at the checkpoint's original global step.
config.resume_optimizer_state = False
config.resume_rebase_lr_scheduler = True

# Phase 1 only asks whether the original curriculum can reach 25/25 with
# finite parameters.  The old averaged auxiliary losses can overflow on rare
# internal tails before the final embedding does, so defer them to recovery.
config.herpn_range_loss_weight = 0.0
config.herpn_distill_loss_weight = 0.0
# The legacy group-boundary recalibration path is intentionally fail-fast on
# any non-finite embedding.  Disable it during this conversion-only pass;
# ordinary training-mode BN updates continue, and phase 2 will recalibrate the
# final 25/25 graph with an explicit finite-batch policy.
config.herpn_bn_recalibration_batches = 0

# Quarantine bad training batches instead of applying NaN/Inf updates.  These
# generous ceilings are deliberately non-acceptance gates; every skip remains
# logged.  A later recovery phase must evaluate skip rates and strict IJB-C.
config.max_nonfinite_embedding_skips = 1_000_000
config.max_nonfinite_loss_skips = 1_000_000
config.skip_nonfinite_gradients = True
config.max_nonfinite_gradient_skips = 1_000_000

# Validation remains diagnostic and non-fatal in this conversion-only phase.
config.fail_on_nonfinite_val = False
config.max_validation_embedding_abs = None

# Preserve an epoch-by-epoch finite recovery boundary through the final
# conversion at epoch 20 and the original hold period through epoch 24.
config.save_all_states = True
config.checkpoint_interval_epochs = 1
config.save_epoch_models = True
config.epoch_model_interval = 1

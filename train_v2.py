import argparse
import heapq
import json
import logging
import math
import os
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from backbones import get_model
from dataset import DatasetWithIndex, get_dataloader
from losses import build_margin_loss
from lr_scheduler import CosineLRWarmup, PolynomialLRWarmup
from partial_fc_v2 import PartialFC_V2
from torch import distributed
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler as TorchDistributedSampler
from torch.utils.tensorboard import SummaryWriter
from utils.utils_callbacks import CallBackLogging, CallBackVerification
from utils.utils_config import get_config
from utils.utils_distributed_sampler import setup_seed
from utils.utils_logging import AverageMeter, init_logging
from utils.utils_layerwise_poly import (
    activation_range_is_contained,
    calibrated_conversion_prefix,
    causally_calibrate_polynomial_group,
    fractional_group_starts_crossed,
    load_tail_replay_manifests,
    merge_tail_replay_indices,
    pending_group_requires_calibration,
    prioritized_tail_replay_indices,
)
from utils.utils_optimizer import (
    clip_grad_norm_stable,
    nonfinite_gradient_diagnostics,
    nonfinite_gradient_tensor_count,
    select_gradient_clip_parameters,
    split_weight_decay_parameters,
    temporary_optimizer_lr_scale,
)
from utils.utils_pillar import (
    pillar_regularization_at_epoch,
    pillar_task_loss_weight_at_epoch,
    pillar_validation_is_strict_at_epoch,
)
from utils.utils_tail_recovery import (
    load_fixed_tail_replay_indices,
    load_fixed_tail_replay_orientations,
)
from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import fp16_compress_hook

assert torch.__version__ >= "1.12.0", "In order to enjoy the features of the new torch, \
we have upgraded the torch to 1.12.0. torch before than 1.12.0 may not work in the future."

try:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    distributed.init_process_group("nccl")
except KeyError:
    rank = 0
    local_rank = 0
    world_size = 1
    distributed.init_process_group(
        backend="nccl",
        init_method="tcp://127.0.0.1:12584",
        rank=rank,
        world_size=world_size,
    )


def check_finite_gradients(module: torch.nn.Module, name: str, global_step: int):
    for param_name, param in module.named_parameters():
        if param.grad is not None and not torch.isfinite(param.grad).all():
            grad = param.grad.detach()
            finite = grad[torch.isfinite(grad)]
            finite_max = finite.abs().max().item() if finite.numel() > 0 else float('nan')
            raise FloatingPointError(
                f"Non-finite gradient in {name}.{param_name} at global_step={global_step}; "
                f"shape={tuple(grad.shape)}, finite_abs_max={finite_max}"
            )


def set_prepbn_progress(module: torch.nn.Module, current_step: int, total_steps: int):
    for submodule in module.modules():
        if hasattr(submodule, "set_progress"):
            submodule.set_progress(current_step, total_steps)


def prepbn_transition_complete(current_step: int, total_steps: int) -> bool:
    return total_steps <= 0 or current_step >= total_steps


def frozen_std_group_steps(start_epoch, gap_steps, steps_per_epoch,
                           group_count):
    """Build ordered transition-start steps for frozen-std sites."""
    start_epoch = float(start_epoch)
    gap_steps = int(gap_steps)
    steps_per_epoch = int(steps_per_epoch)
    group_count = int(group_count)
    if start_epoch < 0.0:
        raise ValueError("frozen_std_start_epoch must be non-negative")
    if gap_steps <= 0:
        raise ValueError("frozen_std_group_gap_steps must be positive")
    if steps_per_epoch <= 0:
        raise ValueError("steps_per_epoch must be positive")
    if group_count < 0:
        raise ValueError("group_count must be non-negative")
    # ``global_step`` is one-based inside the batch loop.  Adding one makes an
    # integer epoch boundary switch on the first batch of the following epoch.
    first_step = int(round(start_epoch * steps_per_epoch)) + 1
    return tuple(first_step + index * gap_steps
                 for index in range(group_count))


def simple_gate_blends_at_epoch(epoch_value, group_epochs, transition_epochs):
    """Return one GELU-to-gate blend for each ordered conversion group."""
    starts = tuple(float(value) for value in group_epochs)
    transition_epochs = float(transition_epochs)
    if not starts:
        return ()
    if transition_epochs <= 0:
        raise ValueError("simple_gate_transition_epochs must be positive")
    if any(right < left + transition_epochs
           for left, right in zip(starts, starts[1:])):
        raise ValueError(
            "SimpleGate transitions must be ordered and non-overlapping")
    return tuple(
        min(max((float(epoch_value) - start) / transition_epochs, 0.0), 1.0)
        for start in starts
    )


def set_simple_gate_instrumentation(module: torch.nn.Module, enabled: bool,
                                    gradient_scale: float = 1.0):
    setter = getattr(module, "set_simple_gate_instrumentation", None)
    if setter is not None:
        setter(enabled, gradient_scale=gradient_scale)


def collect_simple_gate_stats(module: torch.nn.Module):
    collector = getattr(module, "simple_gate_range_stats", None)
    return collector() if collector is not None else {}


def set_simple_gate_blends(module: torch.nn.Module, blends):
    setter = getattr(module, "set_simple_gate_blends", None)
    if setter is not None:
        setter(blends)


def serialize_simple_gate_stats(stats):
    return {
        layer_name: {
            metric: float(value.detach().item())
            for metric, value in layer_stats.items()
        }
        for layer_name, layer_stats in stats.items()
    }


def log_simple_gate_stats(stats, global_step, summary_writer=None,
                          wandb_logger=None, prefix="SimpleGate"):
    """Log every gate plus compact network-wide stability summaries."""
    if not stats:
        return {}
    serialized = serialize_simple_gate_stats(stats)
    for layer_name, layer_stats in serialized.items():
        tensorboard_layer = layer_name.replace(".", "/")
        if summary_writer is not None:
            for metric, value in layer_stats.items():
                summary_writer.add_scalar(
                    f"{prefix}/{tensorboard_layer}/{metric}", value, global_step)

    product_absmax = max(
        values["product_absmax"] for values in serialized.values())
    product_p999 = max(
        values["product_p999"] for values in serialized.values())
    outside_fraction = sum(
        values["product_outside_fraction"] for values in serialized.values()
    ) / len(serialized)
    gradient_absmax = max(
        (values.get("gradient_absmax", 0.0) for values in serialized.values()),
        default=0.0,
    )
    summary = {
        "product_absmax": product_absmax,
        "product_p999": product_p999,
        "product_outside_fraction": outside_fraction,
        "gradient_absmax": gradient_absmax,
    }
    if summary_writer is not None:
        for metric, value in summary.items():
            summary_writer.add_scalar(f"{prefix}/Summary/{metric}", value, global_step)
    if wandb_logger:
        wandb_logger.log({
            f"{prefix}/Summary/{metric}": value for metric, value in summary.items()
        })

    worst_layers = sorted(
        serialized.items(),
        key=lambda item: item[1]["product_absmax"],
        reverse=True,
    )[:3]
    logging.info(
        "[%s][%d] product_absmax=%.6g p99.9=%.6g outside=%.6g "
        "gradient_absmax=%.6g worst=%s",
        prefix,
        global_step,
        product_absmax,
        product_p999,
        outside_fraction,
        gradient_absmax,
        ", ".join(
            f"{name}:{values['product_absmax']:.6g}"
            for name, values in worst_layers
        ),
    )
    return serialized


def log_nf_range_stats(stats, global_step, summary_writer=None,
                       wandb_logger=None):
    """Log normalization-free block ranges and return scalar snapshots."""
    if not stats:
        return {}
    serialized = {
        block_name: {
            metric: float(value.detach().item())
            for metric, value in block_stats.items()
        }
        for block_name, block_stats in stats.items() if block_stats
    }
    if not serialized:
        return {}
    for block_name, block_stats in serialized.items():
        if summary_writer is not None:
            for metric, value in block_stats.items():
                summary_writer.add_scalar(
                    f"NFRange/{block_name}/{metric}", value, global_step)
    product_absmax = max(
        values["product_absmax"] for values in serialized.values())
    product_p999 = max(
        values["product_p999"] for values in serialized.values())
    output_rms = max(values["output_rms"] for values in serialized.values())
    summary = {
        "product_absmax": product_absmax,
        "product_p999": product_p999,
        "output_rms_max": output_rms,
    }
    modulation_progresses = [
        values["modulation_progress"]
        for values in serialized.values()
        if "modulation_progress" in values
    ]
    effective_modulations = [
        abs(values["effective_modulation"])
        for values in serialized.values()
        if "effective_modulation" in values
    ]
    if modulation_progresses:
        summary["modulation_progress_sum"] = sum(modulation_progresses)
        summary["effective_modulation_absmax"] = max(
            effective_modulations, default=0.0)
    if summary_writer is not None:
        for metric, value in summary.items():
            summary_writer.add_scalar(
                f"NFRange/Summary/{metric}", value, global_step)
    if wandb_logger:
        wandb_logger.log({
            f"NFRange/Summary/{metric}": value
            for metric, value in summary.items()
        })
    worst = max(
        serialized.items(), key=lambda item: item[1]["product_absmax"])
    worst_stats = worst[1]
    logging.info(
        "[NFRange][%d] product_absmax=%.6g p99.9=%.6g "
        "output_rms_max=%.6g modulation_progress_sum=%.3f "
        "effective_modulation_absmax=%.6g worst=%s "
        "u_absmax=%.6g v_absmax=%.6g modulator_absmax=%.6g "
        "alpha=%.6g input_gain=%.6g",
        global_step, product_absmax, product_p999, output_rms,
        summary.get("modulation_progress_sum", 0.0),
        summary.get("effective_modulation_absmax", 0.0),
        worst[0],
        worst_stats.get("operand_u_absmax", float("nan")),
        worst_stats.get("operand_v_absmax", float("nan")),
        worst_stats.get("modulator_absmax", float("nan")),
        worst_stats.get("alpha", float("nan")),
        worst_stats.get("input_gain", float("nan")),
    )
    return serialized


def load_embedding_teacher_checkpoint(model, checkpoint_path):
    """Strictly load common backbone checkpoint layouts into a teacher."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise TypeError(
            f"Teacher checkpoint must be a dict, got {type(checkpoint)!r}")
    for key in ("state_dict_backbone", "state_dict", "model"):
        nested = checkpoint.get(key)
        if isinstance(nested, dict):
            checkpoint = nested
            break
    state_dict = {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in checkpoint.items()
    }
    model.load_state_dict(state_dict, strict=True)


def begin_batchnorm_recalibration(module: torch.nn.Module, reset=True):
    """Use the final eval graph while updating only BatchNorm statistics."""
    batchnorm_state = [
        (submodule, submodule.training, submodule.momentum)
        for submodule in module.modules()
        if isinstance(submodule, nn.modules.batchnorm._BatchNorm)
    ]
    state = {
        "model_training": module.training,
        "batchnorm": batchnorm_state,
    }
    module.eval()
    for submodule, _, _ in batchnorm_state:
        if reset:
            submodule.reset_running_stats()
        submodule.momentum = None
        submodule.train()
    return state


def end_batchnorm_recalibration(module: torch.nn.Module, state):
    module.train(state["model_training"])
    for submodule, was_training, momentum in state["batchnorm"]:
        submodule.momentum = momentum
        submodule.train(was_training)


def freeze_batchnorm_for_training(module: torch.nn.Module, *, affine=False):
    """Keep checkpoint BatchNorm statistics fixed during optimization.

    Calling ``Module.train()`` recursively re-enables every BatchNorm layer,
    including after verification callbacks.  This helper is therefore safe to
    call before each training forward.  It changes only module mode unless
    ``affine`` is requested, in which case gamma and beta are frozen too.
    """
    count = 0
    for submodule in module.modules():
        if not isinstance(submodule, nn.modules.batchnorm._BatchNorm):
            continue
        submodule.eval()
        if affine:
            if submodule.weight is not None:
                submodule.weight.requires_grad_(False)
            if submodule.bias is not None:
                submodule.bias.requires_grad_(False)
        count += 1
    return count


def snapshot_batchnorm_running_stats(
        module: torch.nn.Module, *, before_activation=None):
    """Copy mutable BN buffers so a rejected forward can be rolled back."""
    snapshots = []
    found_boundary = before_activation is None
    for name, submodule in module.named_modules():
        if before_activation is not None and name == before_activation:
            found_boundary = True
            break
        if not isinstance(
                submodule, nn.modules.batchnorm._BatchNorm):
            continue
        snapshots.append((
            submodule,
            (submodule.running_mean.detach().clone()
             if submodule.running_mean is not None else None),
            (submodule.running_var.detach().clone()
             if submodule.running_var is not None else None),
            (submodule.num_batches_tracked.detach().clone()
             if submodule.num_batches_tracked is not None else None),
        ))
    if not found_boundary:
        raise ValueError(
            "Unknown activation for upstream BatchNorm snapshot: "
            f"{before_activation}")
    return snapshots


@torch.no_grad()
def restore_batchnorm_running_stats(snapshots):
    for submodule, running_mean, running_var, num_batches_tracked in snapshots:
        if running_mean is not None:
            submodule.running_mean.copy_(running_mean)
        if running_var is not None:
            submodule.running_var.copy_(running_var)
        if num_batches_tracked is not None:
            submodule.num_batches_tracked.copy_(num_batches_tracked)


def merge_cumulative_batchnorm_stats(batchnorm_state):
    """Merge cumulative BN statistics from every distributed data shard."""
    if not distributed.is_available() or not distributed.is_initialized():
        return
    if distributed.get_world_size() <= 1:
        return

    for submodule, _, _ in batchnorm_state:
        if (submodule.running_mean is None
                or submodule.running_var is None
                or submodule.num_batches_tracked is None):
            continue
        if isinstance(submodule, torch.nn.SyncBatchNorm):
            # SyncBatchNorm has already used global batch moments on every
            # forward. Preserve its batch count and make rank zero's identical
            # buffers authoritative instead of summing num_batches_tracked.
            distributed.broadcast(submodule.running_mean, src=0)
            distributed.broadcast(submodule.running_var, src=0)
            distributed.broadcast(submodule.num_batches_tracked, src=0)
            continue
        local_batches = submodule.num_batches_tracked.detach().clone()
        total_batches = local_batches.clone()
        distributed.all_reduce(total_batches, op=distributed.ReduceOp.SUM)
        if total_batches.item() == 0:
            continue

        weight = local_batches.to(
            device=submodule.running_mean.device,
            dtype=submodule.running_mean.dtype,
        )
        for running_stat in (
                submodule.running_mean, submodule.running_var):
            weighted_stat = running_stat * weight
            distributed.all_reduce(
                weighted_stat, op=distributed.ReduceOp.SUM)
            running_stat.copy_(
                weighted_stat
                / total_batches.to(dtype=running_stat.dtype))
        submodule.num_batches_tracked.copy_(total_batches)


@torch.no_grad()
def recalibrate_prepbn_batchnorm(backbone, train_loader, num_epochs, start_epoch,
                                dali=False):
    """Reset and refresh BN statistics through the fully converted RepBN path."""
    if num_epochs <= 0:
        return 0
    module = backbone.module
    state = begin_batchnorm_recalibration(module, reset=True)
    completed = 0
    try:
        for stat_epoch in range(num_epochs):
            if isinstance(train_loader, DataLoader):
                sampler = train_loader.sampler
                if hasattr(sampler, "set_epoch"):
                    sampler.set_epoch(start_epoch + stat_epoch)
            for img, _ in train_loader:
                embeddings = backbone(img)
                if not torch.isfinite(embeddings).all():
                    raise FloatingPointError(
                        "Non-finite embeddings during final RepBatchNorm recalibration")
                completed += 1
            if dali:
                train_loader.reset()
    finally:
        end_batchnorm_recalibration(module, state)
    if completed == 0:
        raise RuntimeError("RepBatchNorm recalibration received no batches")
    return completed


@torch.no_grad()
def calibrate_affine_normalization(backbone, train_loader, num_batches,
                                   ridge=1e-6, dali=False,
                                   group_index=None):
    """Fit fixed channel affine maps to the warm-start LayerNorm outputs."""
    if num_batches <= 0:
        raise ValueError("affine_calibration_batches must be positive")
    begin = getattr(backbone, "begin_affine_calibration", None)
    finish = getattr(backbone, "finish_affine_calibration", None)
    if begin is None or finish is None:
        raise TypeError(
            "Affine calibration requested for a backbone without calibration hooks")

    was_training = backbone.training
    device = next(backbone.parameters()).device
    backbone.eval()
    begin(group_index=group_index)
    completed = 0
    try:
        for img, _ in train_loader:
            if img.device != device:
                img = img.to(device=device, non_blocking=True)
            embeddings = backbone(img)
            if not torch.isfinite(embeddings).all():
                raise FloatingPointError(
                    "Non-finite embeddings during affine normalization calibration")
            completed += 1
            if completed >= num_batches:
                break
        if completed == 0:
            raise RuntimeError(
                "Affine normalization calibration received no batches")
        diagnostics = finish(ridge=ridge, distributed=True)
    finally:
        backbone.train(was_training)
        if dali:
            train_loader.reset()
    return completed, diagnostics


@torch.no_grad()
def profile_simple_gate_ranges(backbone, train_loader, num_batches, dali=False):
    """Profile the final eval/RepBN graph over representative training images."""
    module = backbone.module
    if num_batches <= 0 or not hasattr(module, "simple_gate_range_stats"):
        return {}
    was_training = module.training
    module.eval()
    accumulated = {}
    completed = 0
    try:
        for img, _ in train_loader:
            set_simple_gate_instrumentation(module, True)
            embeddings = backbone(img)
            if not torch.isfinite(embeddings).all():
                raise FloatingPointError(
                    "Non-finite embeddings during final SimpleGate profiling")
            batch_stats = serialize_simple_gate_stats(
                collect_simple_gate_stats(module))
            for layer_name, layer_stats in batch_stats.items():
                target = accumulated.setdefault(
                    layer_name,
                    {metric: [] for metric in layer_stats},
                )
                for metric, value in layer_stats.items():
                    target.setdefault(metric, []).append(value)
            completed += 1
            if completed >= num_batches:
                break
    finally:
        set_simple_gate_instrumentation(module, False)
        module.train(was_training)
        if dali:
            train_loader.reset()
    if completed == 0:
        raise RuntimeError("Final SimpleGate profiling received no batches")

    max_metrics = {
        "operand1_absmax", "operand2_absmax", "product_absmax",
    }
    min_metrics = {"finite", "gradient_finite"}
    result = {}
    profile_device = next(module.parameters()).device
    for layer_name, layer_stats in accumulated.items():
        result[layer_name] = {}
        for metric, values in layer_stats.items():
            if metric in max_metrics:
                reduced = max(values)
            elif metric in min_metrics:
                reduced = min(values)
            else:
                reduced = sum(values) / len(values)
            if distributed.is_initialized() and distributed.get_world_size() > 1:
                reduced_tensor = torch.tensor(
                    reduced, device=profile_device)
                if metric in max_metrics:
                    op = distributed.ReduceOp.MAX
                elif metric in min_metrics:
                    op = distributed.ReduceOp.MIN
                else:
                    op = distributed.ReduceOp.SUM
                distributed.all_reduce(reduced_tensor, op=op)
                reduced = float(reduced_tensor.item())
                if op == distributed.ReduceOp.SUM:
                    reduced /= distributed.get_world_size()
            result[layer_name][metric] = reduced
    total_profile_batches = completed
    if distributed.is_initialized() and distributed.get_world_size() > 1:
        completed_tensor = torch.tensor(
            completed, device=profile_device, dtype=torch.long)
        distributed.all_reduce(completed_tensor, op=distributed.ReduceOp.SUM)
        total_profile_batches = int(completed_tensor.item())
    result["_profile"] = {
        "num_batches_across_ranks": total_profile_batches,
        "absmax_reduction": "maximum across batches and ranks",
        "other_metric_reduction": "mean across batches and ranks",
    }
    return result


def herpn_progress_at_epoch(epoch_value, stage_epochs, transition_epochs):
    if not stage_epochs:
        return 5.0
    transition_epochs = float(transition_epochs)
    if transition_epochs <= 0:
        raise ValueError("herpn_transition_epochs must be positive")
    starts = tuple(float(value) for value in stage_epochs)
    if len(starts) != 5:
        raise ValueError("herpn_stage_epochs must contain stem/layer1/layer2/layer3/layer4 starts")
    if any(right < left + transition_epochs for left, right in zip(starts, starts[1:])):
        raise ValueError("HerPN stage transitions must be ordered and non-overlapping")
    return sum(min(max((float(epoch_value) - start) / transition_epochs, 0.0), 1.0)
               for start in starts)


def precise_relu_progress_at_epoch(epoch_value, stage_epochs,
                                   transition_epochs):
    """Return Alpha10-to-polynomial-student curriculum progress.

    Each start transitions the whole network to the next independently fitted
    ReLU polynomial.  Integer progress selects a single polynomial; fractional
    progress blends only the adjacent pair during plaintext training.
    """
    starts = tuple(float(value) for value in stage_epochs)
    transition_epochs = float(transition_epochs)
    if not starts:
        return 0.0
    if transition_epochs <= 0:
        raise ValueError("precise_relu_transition_epochs must be positive")
    if any(right < left + transition_epochs
           for left, right in zip(starts, starts[1:])):
        raise ValueError(
            "PreciseReLU transitions must be ordered and non-overlapping")
    return sum(
        min(max(
            (float(epoch_value) - start) / transition_epochs, 0.0), 1.0)
        for start in starts
    )


def herpn_group_blends_at_epoch(epoch_value, conversion_groups, group_epochs,
                                transition_epochs):
    groups = tuple(tuple(group) for group in conversion_groups)
    starts = tuple(float(value) for value in group_epochs)
    transition_epochs = float(transition_epochs)
    if len(groups) != len(starts):
        raise ValueError(
            "herpn_conversion_groups and herpn_group_epochs must have equal length")
    if transition_epochs <= 0:
        raise ValueError("herpn_transition_epochs must be positive")
    if any(right < left + transition_epochs
           for left, right in zip(starts, starts[1:])):
        raise ValueError("HerPN conversion groups must be ordered and non-overlapping")

    blends = {}
    for group, start in zip(groups, starts):
        blend = min(max(
            (float(epoch_value) - start) / transition_epochs, 0.0), 1.0)
        for activation_name in group:
            if activation_name in blends:
                raise ValueError(
                    f"HerPN activation appears in multiple groups: {activation_name}")
            blends[activation_name] = blend
    return blends


def layerwise_poly_group_phase_at_epoch(epoch_value, conversion_groups,
                                         group_epochs, transition_epochs):
    """Return ``(phase, group_index)`` for grouped polynomial training."""
    groups = tuple(tuple(group) for group in conversion_groups)
    starts = tuple(float(value) for value in group_epochs)
    transition_epochs = float(transition_epochs)
    if len(groups) != len(starts):
        raise ValueError(
            "herpn_conversion_groups and herpn_group_epochs must have equal length")
    value = float(epoch_value)
    for group_index, start in enumerate(starts):
        if value < start:
            return "local_fit", group_index
        if value < start + transition_epochs:
            return "blend", group_index
    return "final_finetune", None


def retain_only_layerwise_poly_group_gradients(module, activation_names):
    """Freeze the backbone for local fit without changing the DDP graph."""
    selected = getattr(module, "layerwise_poly_parameters", None)
    if selected is None:
        raise ValueError("Backbone does not expose layerwise polynomial parameters")
    allowed = {id(parameter) for parameter in selected(activation_names)}
    retained = 0
    cleared = 0
    for parameter in module.parameters():
        if parameter.grad is None:
            continue
        if id(parameter) in allowed:
            retained += 1
        else:
            parameter.grad = None
            cleared += 1
    return retained, cleared


def retain_layerwise_poly_conditioning_gradients(module, activation_names):
    """Keep ordinary backbone gradients but freeze non-active polynomials."""
    selected = getattr(module, "layerwise_poly_parameters", None)
    if selected is None:
        raise ValueError("Backbone does not expose layerwise polynomial parameters")
    all_polynomial = {id(parameter) for parameter in selected()}
    active_polynomial = {
        id(parameter) for parameter in selected(activation_names)
    }
    retained = 0
    cleared = 0
    for parameter in module.parameters():
        if parameter.grad is None:
            continue
        parameter_id = id(parameter)
        if (parameter_id not in all_polynomial
                or parameter_id in active_polynomial):
            retained += 1
        else:
            parameter.grad = None
            cleared += 1
    return retained, cleared


def validate_herpn_conversion_groups(module, conversion_groups):
    expected = {
        name for name, submodule in module.named_modules()
        if (submodule.__class__.__name__ == "ProgressiveHerPNActivation"
            or getattr(
                submodule, "is_progressive_polynomial_activation", False))
    }
    if any(not group for group in conversion_groups):
        raise ValueError("HerPN conversion groups must not be empty")
    scheduled = {name for group in conversion_groups for name in group}
    count = sum(len(group) for group in conversion_groups)
    if count != len(scheduled):
        raise ValueError("An activation occurs more than once in HerPN groups")
    missing = sorted(expected.difference(scheduled))
    unknown = sorted(scheduled.difference(expected))
    if missing or unknown:
        raise ValueError(
            f"Invalid HerPN conversion groups; missing={missing}, unknown={unknown}")


def completed_herpn_groups_from_model(module, conversion_groups):
    """Count the leading groups already fully blended in a loaded checkpoint."""
    activations = dict(module.named_modules())
    completed = 0
    for group in conversion_groups:
        if all(float(activations[name].blend.item()) >= 1.0 for name in group):
            completed += 1
        else:
            break
    return completed


def atomic_torch_save(value, path):
    """Write a checkpoint completely before replacing an existing file."""
    temporary_path = path + ".tmp"
    torch.save(value, temporary_path)
    os.replace(temporary_path, path)


@torch.no_grad()
def recalibrate_herpn_batchnorm(backbone, train_loader, num_batches, global_step,
                                after_activation_name=None):
    module = backbone.module
    if num_batches <= 0 or not hasattr(module, "begin_batchnorm_recalibration"):
        return
    if (after_activation_name is not None
            and hasattr(module, "begin_batchnorm_recalibration_after")):
        state = module.begin_batchnorm_recalibration_after(
            after_activation_name, reset=True)
    else:
        state = module.begin_batchnorm_recalibration(reset=True)
    completed = 0
    device = next(module.parameters()).device
    try:
        for batch in train_loader:
            img = batch[0].to(device=device, non_blocking=True)
            # Bypass DDP so broadcast_buffers cannot overwrite independently
            # accumulated ordinary-BN statistics before every forward.
            # SyncBatchNorm collectives still run through the wrapped module.
            embeddings = module(img)
            if not torch.isfinite(embeddings).all():
                raise FloatingPointError(
                    "Non-finite embeddings during HerPN BatchNorm recalibration "
                    f"at global_step={global_step}, calibration_batch={completed}"
                )
            completed += 1
            if completed >= num_batches:
                break
    finally:
        module.end_batchnorm_recalibration(state)
    if completed == 0:
        raise RuntimeError("HerPN BatchNorm recalibration received no batches")
    merge_cumulative_batchnorm_stats(state["batchnorm"])
    return completed


def _sampled_activation_quantile(values, quantile, max_samples):
    absolute = values.detach().float().abs().reshape(-1)
    if quantile >= 1.0:
        return absolute.amax()
    if absolute.numel() > max_samples:
        stride = math.ceil(absolute.numel() / max_samples)
        absolute = absolute[::stride][:max_samples]
    rank = max(1, math.ceil(quantile * absolute.numel()))
    return absolute.kthvalue(rank).values


def _global_range_source(local_absmax, local_source):
    """Return diagnostics for the rank and activation coordinate owning max."""
    device = local_absmax.device
    rank_value = (
        distributed.get_rank()
        if distributed.is_available() and distributed.is_initialized()
        else 0
    )
    coordinates = list(local_source.get("coordinates", ()))[:4]
    coordinates.extend([-1] * (4 - len(coordinates)))
    row = torch.tensor(
        [
            float(local_absmax.item()),
            rank_value,
            int(local_source.get("batch", -1)),
            int(local_source.get("dataset_index", -1)),
            int(local_source.get("orientation", -1)),
            *coordinates,
        ],
        device=device,
        dtype=torch.float64,
    )
    rows = [row]
    if (distributed.is_available() and distributed.is_initialized()
            and distributed.get_world_size() > 1):
        rows = [torch.empty_like(row) for _ in range(distributed.get_world_size())]
        distributed.all_gather(rows, row)
    winner = max(rows, key=lambda item: (float(item[0]), -int(item[1])))
    return {
        "rank": int(winner[1].item()),
        "batch": int(winner[2].item()),
        "dataset_index": int(winner[3].item()),
        "orientation": int(winner[4].item()),
        "sample": int(winner[5].item()),
        "channel": int(winner[6].item()),
        "height": int(winner[7].item()),
        "width": int(winner[8].item()),
    }


def _global_tail_indices(local_tail, count, device):
    """Merge fixed-size per-rank ``(magnitude, dataset index)`` heaps."""
    if count <= 0:
        return ()
    row = torch.full(
        (count, 2), -1.0, device=device, dtype=torch.float64)
    ordered = sorted(local_tail, reverse=True)[:count]
    for position, (magnitude, dataset_index) in enumerate(ordered):
        row[position, 0] = float(magnitude)
        row[position, 1] = int(dataset_index)
    rows = [row]
    if (distributed.is_available() and distributed.is_initialized()
            and distributed.get_world_size() > 1):
        rows = [
            torch.empty_like(row)
            for _ in range(distributed.get_world_size())
        ]
        distributed.all_gather(rows, row)
    by_index = {}
    for gathered in rows:
        for magnitude, dataset_index in gathered.cpu().tolist():
            dataset_index = int(dataset_index)
            if dataset_index < 0:
                continue
            by_index[dataset_index] = max(
                float(magnitude), by_index.get(dataset_index, float("-inf")))
    return tuple(
        index for index, _ in sorted(
            by_index.items(), key=lambda item: (-item[1], item[0]))[:count]
    )


@torch.no_grad()
def calibrate_layerwise_poly_input_scales(
        backbone, train_loader, activation_names, num_batches, margin,
        min_scale, global_step, dali=False, *, quantile=1.0,
        quantile_samples=65536, holdout_fraction=0.0,
        max_tail_ratio=0.0, max_scale_growth=0.0, max_input_scale=0.0,
        allow_recalibration=False, allow_provisional_tail=False,
        enforce_tail_scale_floor=False, tail_scale_floor_margin=1.0,
        max_tail_scale_expansion=0.0, progress_interval_batches=0,
        require_full_containment=False, preserve_existing_scale=False,
        tail_topk=0):
    """Fit separate intervals for one activation group in one loader pass.

    All requested activations are observed on the same current eval graph.
    Scale updates are atomic: if any member violates a hard safety limit, no
    member is updated. A provisional pass may accept only tail-ratio and
    same-stage scale-growth violations so upstream layers can condition the
    pending inputs while its blend remains zero. ``num_batches=0`` consumes
    the complete representative loader shard on every rank.
    """
    if num_batches < 0:
        raise ValueError("layerwise_poly_range_calibration_batches must be >= 0")
    margin = float(margin)
    min_scale = float(min_scale)
    if margin < 1.0:
        raise ValueError("layerwise_poly_range_margin must be at least 1")
    if min_scale <= 0.0:
        raise ValueError("layerwise_poly_min_scale must be positive")
    quantile = float(quantile)
    holdout_fraction = float(holdout_fraction)
    max_tail_ratio = float(max_tail_ratio)
    max_scale_growth = float(max_scale_growth)
    max_input_scale = float(max_input_scale)
    tail_scale_floor_margin = float(tail_scale_floor_margin)
    max_tail_scale_expansion = float(max_tail_scale_expansion)
    progress_interval_batches = int(progress_interval_batches)
    require_full_containment = bool(require_full_containment)
    preserve_existing_scale = bool(preserve_existing_scale)
    tail_topk = int(tail_topk)
    if not 0.0 < quantile <= 1.0:
        raise ValueError("layerwise_poly_range_quantile must be in (0, 1]")
    if quantile_samples <= 0:
        raise ValueError("layerwise_poly_quantile_samples must be positive")
    if not 0.0 <= holdout_fraction < 0.5:
        raise ValueError(
            "layerwise_poly_range_holdout_fraction must be in [0, 0.5)")
    if min(max_tail_ratio, max_scale_growth, max_input_scale) < 0.0:
        raise ValueError("Layerwise polynomial safety limits must be non-negative")
    if tail_scale_floor_margin < 1.0:
        raise ValueError(
            "layerwise_poly_tail_scale_floor_margin must be at least 1")
    if (max_tail_scale_expansion < 0.0
            or 0.0 < max_tail_scale_expansion < 1.0):
        raise ValueError(
            "layerwise_poly_max_tail_scale_expansion must be zero or at least 1")
    if enforce_tail_scale_floor and max_tail_ratio <= 0.0:
        raise ValueError(
            "Tail scale flooring requires layerwise_poly_max_tail_ratio > 0")
    if progress_interval_batches < 0:
        raise ValueError("Calibration progress interval must be non-negative")
    if tail_topk < 0:
        raise ValueError("tail_topk must be non-negative")
    if preserve_existing_scale and not allow_recalibration:
        raise ValueError(
            "preserve_existing_scale requires allow_recalibration")

    module = backbone.module
    activation_names = tuple(activation_names)
    if not activation_names:
        raise ValueError("Layerwise polynomial calibration group is empty")
    if len(set(activation_names)) != len(activation_names):
        raise ValueError("Layerwise polynomial calibration names must be unique")
    activations = dict(module.named_modules())
    model_order = list(module.layerwise_poly_activation_names())
    try:
        requested_indices = [model_order.index(name) for name in activation_names]
    except ValueError as error:
        raise ValueError(
            f"Unknown layerwise polynomial activation in {activation_names}") from error
    if requested_indices != sorted(requested_indices):
        raise ValueError(
            "Layerwise polynomial calibration group must follow model order")
    for activation_name in activation_names:
        activation = activations[activation_name]
        if not getattr(activation, "is_layerwise_rescaled_polynomial", False):
            raise ValueError(
                f"Unknown layerwise polynomial activation: {activation_name}")
        if (getattr(activation, "_scale_is_calibrated", False)
                and not allow_recalibration):
            raise RuntimeError(
                f"Activation interval is already calibrated: {activation_name}")

    device = next(module.parameters()).device
    local = {}
    for activation_name in activation_names:
        absmax = torch.zeros((), device=device, dtype=torch.float32)
        local[activation_name] = {
            "absmax": absmax,
            "calibration_absmax": torch.zeros_like(absmax),
            "holdout_absmax": torch.zeros_like(absmax),
            "quantile": torch.zeros_like(absmax),
            "source": {
                "batch": -1,
                "dataset_index": -1,
                "orientation": -1,
                "coordinates": (),
            },
            "tail": [],
        }
    completed = 0
    calibration_batches = 0
    holdout_batches = 0
    current_is_holdout = False
    current_dataset_indices = None
    current_orientations = None
    holdout_stride = (
        max(2, round(1.0 / holdout_fraction))
        if holdout_fraction > 0.0 else 0
    )

    def make_capture_input(activation_name):
        state = local[activation_name]

        def capture_input(_, inputs):
            if not inputs or not torch.is_tensor(inputs[0]):
                raise RuntimeError(
                    f"Activation {activation_name} received no tensor input")
            values = inputs[0].detach().float()
            if not torch.isfinite(values).all():
                raise FloatingPointError(
                    "Non-finite activation input during layerwise interval "
                    f"calibration for {activation_name} at "
                    f"global_step={global_step}")
            absolute = values.abs()
            sample_absmax = absolute.flatten(1).amax(dim=1)
            batch_absmax, flat_index = absolute.reshape(-1).max(dim=0)
            if batch_absmax > state["absmax"]:
                state["absmax"].copy_(batch_absmax)
                remaining = int(flat_index.item())
                coordinates = [0] * values.ndim
                for dimension in range(values.ndim - 1, -1, -1):
                    coordinates[dimension] = remaining % values.shape[dimension]
                    remaining //= values.shape[dimension]
                state["source"]["batch"] = completed
                state["source"]["dataset_index"] = (
                    int(current_dataset_indices[coordinates[0]])
                    if current_dataset_indices is not None else -1)
                state["source"]["orientation"] = (
                    int(current_orientations[coordinates[0]])
                    if current_orientations is not None else -1)
                state["source"]["coordinates"] = tuple(coordinates)
            if tail_topk > 0:
                if current_dataset_indices is None:
                    raise RuntimeError(
                        "tail_topk calibration requires a loader that returns "
                        "stable dataset indices")
                candidate_count = min(tail_topk, sample_absmax.numel())
                magnitudes, positions = sample_absmax.topk(candidate_count)
                for magnitude, position in zip(
                        magnitudes.cpu().tolist(), positions.cpu().tolist()):
                    item = (
                        float(magnitude),
                        int(current_dataset_indices[int(position)]),
                    )
                    if len(state["tail"]) < tail_topk:
                        heapq.heappush(state["tail"], item)
                    elif item > state["tail"][0]:
                        heapq.heapreplace(state["tail"], item)
            if current_is_holdout:
                state["holdout_absmax"].copy_(torch.maximum(
                    state["holdout_absmax"], batch_absmax))
            else:
                state["calibration_absmax"].copy_(torch.maximum(
                    state["calibration_absmax"], batch_absmax))
                batch_quantile = _sampled_activation_quantile(
                    values, quantile, int(quantile_samples))
                state["quantile"].copy_(torch.maximum(
                    state["quantile"], batch_quantile))

        return capture_input

    training_states = [
        (submodule, submodule.training) for submodule in module.modules()
    ]
    handles = [
        activations[name].register_forward_pre_hook(make_capture_input(name))
        for name in activation_names
    ]
    module.eval()
    try:
        for batch in train_loader:
            if not isinstance(batch, (tuple, list)) or len(batch) < 2:
                raise ValueError(
                    "Layerwise calibration loader must return image and label")
            img = batch[0]
            current_dataset_indices = (
                batch[2].detach().cpu().tolist()
                if len(batch) >= 3 else None)
            current_orientations = (
                batch[3].detach().cpu().tolist()
                if len(batch) >= 4 else None)
            if img.device != device:
                img = img.to(device=device, non_blocking=True)
            current_is_holdout = (
                holdout_stride > 0 and completed % holdout_stride == holdout_stride - 1)
            embeddings = module(img)
            if not torch.isfinite(embeddings).all():
                raise FloatingPointError(
                    "Non-finite embeddings during layerwise interval "
                    f"calibration for group {activation_names} at "
                    f"global_step={global_step}, batch={completed}")
            if current_is_holdout:
                holdout_batches += 1
            else:
                calibration_batches += 1
            completed += 1
            if (rank == 0 and progress_interval_batches > 0
                    and completed % progress_interval_batches == 0):
                logging.info(
                    "Layerwise polynomial calibration progress: group=%s "
                    "batches/rank=%d",
                    ", ".join(activation_names), completed)
            if num_batches > 0 and completed >= num_batches:
                break
    finally:
        for handle in handles:
            handle.remove()
        for submodule, was_training in training_states:
            submodule.train(was_training)
        if dali:
            train_loader.reset()

    if completed == 0:
        raise RuntimeError(
            f"Interval calibration for {activation_names} received no batches")
    if calibration_batches == 0:
        raise RuntimeError(
            f"Interval calibration for {activation_names} has no fit batches")

    results = []
    proposed_scales = {}
    unsafe = []
    for activation_name in activation_names:
        state = local[activation_name]
        source = _global_range_source(state["absmax"], state["source"])
        global_values = torch.stack((
            state["absmax"],
            state["calibration_absmax"],
            state["holdout_absmax"],
            state["quantile"],
        ))
        if distributed.is_available() and distributed.is_initialized():
            distributed.all_reduce(global_values, op=distributed.ReduceOp.MAX)
        observed_absmax = float(global_values[0].item())
        calibration_absmax = float(global_values[1].item())
        holdout_absmax = float(global_values[2].item())
        robust_absmax = float(global_values[3].item())
        robust_scale = max(robust_absmax * margin, min_scale)
        activation = activations[activation_name]
        calibrated_scale = (
            float(activation.input_scale.item())
            if (preserve_existing_scale
                and getattr(activation, "_scale_is_calibrated", False))
            else robust_scale)
        tail_scale_floor = None
        tail_scale_expansion = 1.0
        tail_floor_applied = False
        tail_floor_hard_violation = None
        if enforce_tail_scale_floor and observed_absmax > 0.0:
            tail_scale_floor = (
                observed_absmax / max_tail_ratio * tail_scale_floor_margin)
            tail_scale_expansion = tail_scale_floor / robust_scale
            if (max_tail_scale_expansion > 0.0
                    and tail_scale_expansion > max_tail_scale_expansion):
                tail_floor_hard_violation = (
                    "tail_scale_expansion="
                    f"{tail_scale_expansion:.7g}>"
                    f"{max_tail_scale_expansion:.7g}")
            elif tail_scale_floor > calibrated_scale:
                calibrated_scale = tail_scale_floor
                tail_floor_applied = True

        activation_index = model_order.index(activation_name)
        previous_scale = None
        if activation_index > 0:
            previous_name = model_order[activation_index - 1]
            if previous_name in proposed_scales:
                previous_scale = proposed_scales[previous_name]
            else:
                previous = activations[previous_name]
                if getattr(previous, "_scale_is_calibrated", False):
                    previous_scale = float(previous.input_scale.item())
        previous_activation = (
            activations[model_order[activation_index - 1]]
            if activation_index > 0 else None
        )
        same_stage_as_previous = (
            previous_activation is not None
            and getattr(previous_activation, "stage_index", None)
            == getattr(activations[activation_name], "stage_index", None)
        )
        # Adjacent iResNet stages intentionally change width and spatial
        # resolution. Their scales are not comparable, so scale-growth is only
        # a useful runaway heuristic within a stage.
        scale_growth = (
            calibrated_scale / previous_scale
            if (same_stage_as_previous and previous_scale is not None
                and previous_scale > 0.0)
            else None
        )
        tail_ratio = observed_absmax / max(calibrated_scale, min_scale)
        provisional_violations = []
        hard_violations = []
        if tail_floor_hard_violation is not None:
            hard_violations.append(tail_floor_hard_violation)
        if max_tail_ratio > 0.0 and tail_ratio > max_tail_ratio:
            provisional_violations.append(
                f"tail_ratio={tail_ratio:.7g}>{max_tail_ratio:.7g}")
        if (require_full_containment
                and not activation_range_is_contained(
                    observed_absmax, calibrated_scale)):
            provisional_violations.append(
                f"containment_ratio={tail_ratio:.7g}>1")
        if (max_scale_growth > 0.0 and scale_growth is not None
                and scale_growth > max_scale_growth):
            provisional_violations.append(
                f"scale_growth={scale_growth:.7g}>{max_scale_growth:.7g}")
        if max_input_scale > 0.0 and calibrated_scale > max_input_scale:
            hard_violations.append(
                f"scale={calibrated_scale:.7g}>{max_input_scale:.7g}")
        violations = hard_violations + provisional_violations
        result = {
            "activation": activation_name,
            "observed_absmax": observed_absmax,
            "calibration_absmax": calibration_absmax,
            "holdout_absmax": holdout_absmax,
            "robust_absmax": robust_absmax,
            "quantile": quantile,
            "input_scale": calibrated_scale,
            "robust_input_scale": robust_scale,
            "tail_scale_floor": tail_scale_floor,
            "tail_scale_expansion": tail_scale_expansion,
            "tail_floor_applied": tail_floor_applied,
            "previous_scale": previous_scale,
            "scale_growth": scale_growth,
            "same_stage_as_previous": same_stage_as_previous,
            "tail_ratio": tail_ratio,
            "violations": tuple(violations),
            "provisional": bool(
                provisional_violations and allow_provisional_tail),
            "source": source,
            "tail_indices": _global_tail_indices(
                state["tail"], tail_topk, device),
            "margin": margin,
            "batches_per_rank": completed,
            "calibration_batches_per_rank": calibration_batches,
            "holdout_batches_per_rank": holdout_batches,
        }
        results.append(result)
        proposed_scales[activation_name] = calibrated_scale
        if hard_violations or (
                provisional_violations and not allow_provisional_tail):
            unsafe.append((result, violations))

    if unsafe:
        details = []
        for result, violations in unsafe:
            details.append(
                f'{result["activation"]}: ' + ", ".join(violations)
                + f'; robust_q={quantile:.7g}, '
                  f'robust_absmax={result["robust_absmax"]:.7g}, '
                  f'calibration_absmax={result["calibration_absmax"]:.7g}, '
                  f'holdout_absmax={result["holdout_absmax"]:.7g}, '
                  f'observed_absmax={result["observed_absmax"]:.7g}, '
                  f'source={result["source"]}')
        raise FloatingPointError(
            "Unsafe layerwise polynomial interval group at "
            f"global_step={global_step}: " + " | ".join(details))

    for result in results:
        activation = activations[result["activation"]]
        if not (
                preserve_existing_scale
                and getattr(activation, "_scale_is_calibrated", False)):
            module.set_layerwise_poly_input_scale(
                result["activation"], result["input_scale"])
    return results


@torch.no_grad()
def verify_layerwise_poly_group_boundary(
        backbone, train_loader, activation_names, num_batches, global_step,
        max_boundary_abs, dali=False, progress_interval_batches=0):
    """Verify the fully polynomial group at its immediate downstream input.

    The caller temporarily enables the complete group.  For non-final groups
    this profiles the next polynomial activation's input; for the final group
    it profiles embeddings.  This catches finite but catastrophic cascades
    before any training step uses the new blend.
    """
    module = backbone.module
    names = tuple(activation_names)
    model_order = tuple(module.layerwise_poly_activation_names())
    last_index = model_order.index(names[-1])
    boundary_name = (
        model_order[last_index + 1]
        if last_index + 1 < len(model_order) else "embeddings")
    boundary_module = (
        dict(module.named_modules())[boundary_name]
        if boundary_name != "embeddings" else None)
    device = next(module.parameters()).device
    local_absmax = torch.zeros((), device=device, dtype=torch.float32)
    local_nonfinite = torch.zeros((), device=device, dtype=torch.float32)
    completed = 0

    def capture_boundary(_, inputs):
        values = inputs[0].detach().float()
        finite = torch.isfinite(values)
        if not bool(finite.all()):
            local_nonfinite.fill_(1.0)
        if bool(finite.any()):
            local_absmax.copy_(torch.maximum(
                local_absmax, values[finite].abs().amax()))

    training_states = [
        (submodule, submodule.training) for submodule in module.modules()
    ]
    handle = (
        boundary_module.register_forward_pre_hook(capture_boundary)
        if boundary_module is not None else None)
    module.eval()
    try:
        for batch in train_loader:
            if not isinstance(batch, (tuple, list)) or len(batch) < 2:
                raise ValueError(
                    "Causal verification loader must return image and label")
            img = batch[0]
            if img.device != device:
                img = img.to(device=device, non_blocking=True)
            embeddings = module(img)
            if boundary_module is None:
                finite = torch.isfinite(embeddings)
                if not bool(finite.all()):
                    local_nonfinite.fill_(1.0)
                if bool(finite.any()):
                    local_absmax.copy_(torch.maximum(
                        local_absmax,
                        embeddings.detach().float()[finite].abs().amax()))
            completed += 1
            if (rank == 0 and progress_interval_batches > 0
                    and completed % progress_interval_batches == 0):
                logging.info(
                    "Causal polynomial verification progress: group=%s "
                    "boundary=%s batches/rank=%d",
                    ", ".join(names), boundary_name, completed)
            if num_batches > 0 and completed >= num_batches:
                break
    finally:
        if handle is not None:
            handle.remove()
        for submodule, was_training in training_states:
            submodule.train(was_training)
        if dali:
            train_loader.reset()

    if completed == 0:
        raise RuntimeError(
            f"Causal polynomial verification for {names} received no batches")
    diagnostics = torch.stack((local_absmax, local_nonfinite))
    if distributed.is_available() and distributed.is_initialized():
        distributed.all_reduce(diagnostics, op=distributed.ReduceOp.MAX)
    boundary_absmax = float(diagnostics[0].item())
    nonfinite = bool(diagnostics[1].item())
    if nonfinite or (
            max_boundary_abs > 0.0 and boundary_absmax > max_boundary_abs):
        reason = (
            "non-finite values"
            if nonfinite else
            f"absmax={boundary_absmax:.7g}>{max_boundary_abs:.7g}")
        raise FloatingPointError(
            "Unsafe fully polynomial group boundary at "
            f"global_step={global_step}, group={names}, "
            f"boundary={boundary_name}: {reason}")
    return {
        "boundary": boundary_name,
        "absmax": boundary_absmax,
        "batches_per_rank": completed,
    }


@torch.no_grad()
def calibrate_layerwise_poly_input_scale(
        backbone, train_loader, activation_name, num_batches, margin,
        min_scale, global_step, dali=False, **kwargs):
    """Backward-compatible single-activation calibration wrapper."""
    return calibrate_layerwise_poly_input_scales(
        backbone, train_loader, (activation_name,), num_batches, margin,
        min_scale, global_step, dali=dali, **kwargs)[0]


@torch.no_grad()
def recalibrate_batchnorm_batches(backbone, train_loader, num_batches,
                                  global_step, reason):
    """Reset and refresh BN statistics using the current inference graph."""
    if num_batches <= 0:
        return 0
    module = backbone.module
    state = begin_batchnorm_recalibration(module, reset=True)
    completed = 0
    try:
        for img, _ in train_loader:
            # Bypass DDP so broadcast_buffers cannot replace each rank's
            # independently accumulated running statistics before a forward.
            embeddings = module(img)
            if not torch.isfinite(embeddings).all():
                raise FloatingPointError(
                    f"Non-finite embeddings during {reason} BatchNorm "
                    f"recalibration at global_step={global_step}, "
                    f"calibration_batch={completed}")
            completed += 1
            if completed >= num_batches:
                break
    finally:
        end_batchnorm_recalibration(module, state)
    if completed == 0:
        raise RuntimeError(f"{reason} BatchNorm recalibration received no batches")
    merge_cumulative_batchnorm_stats(state["batchnorm"])
    return completed



class CryptoFaceArcFaceHead(nn.Module):
    def __init__(self, embedding_dim: int, num_classes: int, s: float = 64.0, m: float = 0.5):
        super().__init__()
        self.kernel = nn.Parameter(torch.empty(embedding_dim, num_classes))
        self.kernel.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        self.s = s
        self.m = m
        self.eps = 1e-4

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor):
        embeddings = embeddings / torch.norm(embeddings, 2, 1, True)
        kernel = self.kernel / torch.norm(self.kernel, 2, 0, True)
        cosine = (embeddings @ kernel).clamp(-1 + self.eps, 1 - self.eps)

        m_hot = torch.zeros(labels.size(0), cosine.size(1), device=cosine.device, dtype=cosine.dtype)
        m_hot.scatter_(1, labels.reshape(-1, 1), self.m)

        theta = cosine.acos()
        theta_m = torch.clip(theta + m_hot, min=self.eps, max=math.pi - self.eps)
        return theta_m.cos() * self.s


def is_cryptoface_patch_training(cfg):
    return cfg.network == "patch_cnn" and getattr(cfg, "patch_cnn_training", "") == "cryptoface"

def main(args):

    # get config
    cfg = get_config(args.config)
    # global control random seed
    setup_seed(seed=cfg.seed, cuda_deterministic=False)

    torch.cuda.set_device(local_rank)

    os.makedirs(cfg.output, exist_ok=True)
    init_logging(rank, cfg.output)

    summary_writer = (
        SummaryWriter(log_dir=os.path.join(cfg.output, "tensorboard"))
        if rank == 0 and getattr(cfg, "tensorboard", True)
        else None
    )
    
    wandb_logger = None
    if cfg.using_wandb:
        import wandb
        # Sign in to wandb
        try:
            wandb.login(key=cfg.wandb_key)
        except Exception as e:
            print("WandB Key must be provided in config file (base.py).")
            print(f"Config Error: {e}")
        # Initialize wandb
        run_name = datetime.now().strftime("%y%m%d_%H%M") + f"_GPU{rank}"
        run_name = run_name if cfg.suffix_run_name is None else run_name + f"_{cfg.suffix_run_name}"
        try:
            wandb_logger = wandb.init(
                entity = cfg.wandb_entity, 
                project = cfg.wandb_project, 
                sync_tensorboard = True,
                resume=cfg.wandb_resume,
                name = run_name, 
                notes = cfg.notes) if rank == 0 or cfg.wandb_log_all else None
            if wandb_logger:
                wandb_logger.config.update(cfg)
        except Exception as e:
            print("WandB Data (Entity and Project name) must be provided in config file (base.py).")
            print(f"Config Error: {e}")
    train_loader = get_dataloader(
        cfg.rec,
        local_rank,
        cfg.batch_size,
        cfg.dali,
        cfg.dali_aug,
        cfg.seed,
        cfg.num_workers,
        range_augmentation=getattr(cfg, "range_augmentation", None),
    )
    herpn_recalibration_loader = train_loader
    if (not cfg.dali
            and (int(getattr(cfg, "herpn_bn_recalibration_batches", 0)) > 0
                 or int(getattr(
                     cfg, "precise_relu_bn_recalibration_batches", 0)) > 0)):
        # DataLoaderX cannot safely abandon a partially consumed iterator: its
        # background prefetch thread remains blocked on a full queue. HerPN
        # conversion does this after every group, so use a regular loader whose
        # iterator and workers are released after each short calibration pass.
        herpn_recalibration_loader = DataLoader(
            dataset=train_loader.dataset,
            batch_size=cfg.batch_size,
            sampler=train_loader.sampler,
            num_workers=cfg.num_workers,
            pin_memory=True,
            drop_last=True,
            worker_init_fn=train_loader.worker_init_fn,
        )
    affine_calibration_loader = train_loader
    if (cfg.network.startswith("poolformer_fully_gated_affine")
            and not cfg.dali):
        # DataLoaderX owns a background prefetch thread which cannot be safely
        # abandoned after a short calibration pass.  Use a regular loader so
        # each per-group iterator shuts its workers down when calibration ends.
        affine_calibration_loader = DataLoader(
            dataset=train_loader.dataset,
            batch_size=cfg.batch_size,
            sampler=train_loader.sampler,
            num_workers=cfg.num_workers,
            pin_memory=True,
            drop_last=True,
            worker_init_fn=train_loader.worker_init_fn,
        )
    layerwise_poly_range_loader = train_loader
    if (hasattr(cfg, "layerwise_poly_range_calibration_batches")
            and not cfg.dali):
        # A finite calibration pass abandons its iterator after N batches.
        # DataLoaderX would leave its background prefetch thread blocked on a
        # full queue after every initial/strict interval fit, so use a regular
        # loader whose iterator workers are released at the end of each pass.
        indexed_range_dataset = DatasetWithIndex(
            train_loader.dataset,
            both_orientations=bool(getattr(
                cfg, "layerwise_poly_scan_both_orientations", False)),
        )
        indexed_range_sampler = TorchDistributedSampler(
            indexed_range_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            seed=int(cfg.seed),
            drop_last=False,
        )
        layerwise_poly_range_loader = DataLoader(
            dataset=indexed_range_dataset,
            batch_size=cfg.batch_size,
            sampler=indexed_range_sampler,
            num_workers=cfg.num_workers,
            pin_memory=True,
            drop_last=False,
            worker_init_fn=train_loader.worker_init_fn,
        )

    model_kwargs = {
        "dropout": 0.0,
        "fp16": cfg.fp16,
        "num_features": cfg.embedding_size,
    }
    if hasattr(cfg, "arch_config"):
        model_kwargs["arch_config"] = str(cfg.arch_config)

    if (cfg.network.startswith("r")
            and cfg.network.endswith(("_no_relu", "_prelu_herpn"))):
        default_herpn_progress = (
            0.0 if (getattr(cfg, "herpn_conversion_groups", ())
                    or getattr(cfg, "herpn_stage_epochs", ())) else 5.0)
        model_kwargs.update(
            herpn_range_limit=float(getattr(cfg, "herpn_range_limit", 6.0)),
            herpn_bn_eps=float(getattr(cfg, "herpn_bn_eps", 1e-4)),
            herpn_progress=float(getattr(
                cfg, "herpn_initial_progress", default_herpn_progress)),
        )
        if cfg.network.endswith("_no_relu"):
            model_kwargs.update(
                herpn_range_penalty_mode=str(getattr(
                    cfg, "herpn_range_penalty_mode", "legacy")),
                herpn_range_topk_fraction=float(getattr(
                    cfg, "herpn_range_topk_fraction", 0.001)),
                herpn_range_bulk_weight=float(getattr(
                    cfg, "herpn_range_bulk_weight", 0.01)),
                herpn_training_stabilization_limit=getattr(
                    cfg, "herpn_training_stabilization_limit", None),
                herpn_training_stabilization_names=tuple(getattr(
                    cfg, "herpn_training_stabilization_names", ())),
            )
        if cfg.network.endswith("_prelu_herpn"):
            model_kwargs["prelu_herpn_distill_eps"] = float(getattr(
                cfg, "prelu_herpn_distill_eps", 1e-4))
            if hasattr(cfg, "prelu_herpn_layerwise_scale"):
                model_kwargs["prelu_herpn_layerwise_scale"] = bool(
                    cfg.prelu_herpn_layerwise_scale)
                model_kwargs["prelu_herpn_initial_scale"] = float(getattr(
                    cfg, "prelu_herpn_initial_scale", 1.0))
            model_kwargs["prelu_herpn_legacy_prefix"] = int(getattr(
                cfg, "prelu_herpn_legacy_prefix", 0))
            model_kwargs["prelu_herpn_linear_indices"] = tuple(getattr(
                cfg, "prelu_herpn_linear_indices", ()))
            model_kwargs["prelu_herpn_linear_trainable"] = bool(getattr(
                cfg, "prelu_herpn_linear_trainable", True))
    if (cfg.network.startswith("r")
            and cfg.network.endswith("_herpn_residual_scale")):
        model_kwargs.update(
            herpn_range_limit=float(getattr(
                cfg, "herpn_range_limit", 6.0)),
            herpn_bn_eps=float(getattr(cfg, "herpn_bn_eps", 1e-4)),
            residual_scale_init=float(getattr(
                cfg, "residual_scale_init", 1.0 / math.sqrt(24.0))),
            residual_scale_trainable=bool(getattr(
                cfg, "residual_scale_trainable", True)),
        )
    if cfg.network.startswith("r") and cfg.network.endswith("_quadratic"):
        model_kwargs.update(
            quadratic_input_scale=float(getattr(
                cfg, "quadratic_input_scale", 6.0)),
            quadratic_range_limit=float(getattr(
                cfg, "quadratic_range_limit", 6.0)),
            quadratic_abs_init=float(getattr(
                cfg, "quadratic_abs_init",
                1.0 / math.sqrt(2.0 * math.pi))),
            quadratic_progress=float(getattr(
                cfg, "herpn_initial_progress", 0.0)),
        )
    if cfg.network.startswith("r") and cfg.network.endswith("_pillar"):
        model_kwargs.update(
            pillar_approximation_range=float(getattr(
                cfg, "pillar_approximation_range", 5.0)),
            pillar_regularization_range=float(getattr(
                cfg, "pillar_regularization_range", 4.8)),
            pillar_regularization_exponent=int(getattr(
                cfg, "pillar_regularization_exponent", 10)),
            pillar_training_clip=bool(getattr(
                cfg, "pillar_training_clip", True)),
            pillar_penalty_reduction=str(getattr(
                cfg, "pillar_penalty_reduction", "mean")),
            pillar_penalty_tail_cap=getattr(
                cfg, "pillar_penalty_tail_cap", None),
            pillar_input_scale=float(getattr(
                cfg, "pillar_input_scale", 1.0)),
            pillar_input_scale_overrides=dict(getattr(
                cfg, "pillar_input_scale_overrides", {})),
        )
    if (cfg.network.startswith("r")
            and cfg.network.endswith("_layerwise_poly")):
        model_kwargs.update(
            layerwise_poly_degree=int(getattr(
                cfg, "layerwise_poly_degree", 2)),
            layerwise_poly_initial_scale=float(getattr(
                cfg, "layerwise_poly_initial_scale", 1.0)),
            layerwise_poly_distill_eps=float(getattr(
                cfg, "layerwise_poly_distill_eps", 1e-4)),
            layerwise_poly_range_penalty_mode=str(getattr(
                cfg, "layerwise_poly_range_penalty_mode", "legacy")),
            layerwise_poly_range_topk_fraction=float(getattr(
                cfg, "layerwise_poly_range_topk_fraction", 0.25)),
            layerwise_poly_range_bulk_weight=float(getattr(
                cfg, "layerwise_poly_range_bulk_weight", 0.01)),
            layerwise_poly_range_guard_ratio=float(getattr(
                cfg, "layerwise_poly_range_guard_ratio", 1.0)),
            layerwise_poly_progress=float(getattr(
                cfg, "herpn_initial_progress", 0.0)),
        )
    if cfg.network.startswith("r") and "_precise_relu" in cfg.network:
        alpha7_backbone = cfg.network.endswith("_precise_relu_alpha7")
        model_kwargs.update(
            precise_relu_input_scale=float(getattr(
                cfg, "precise_relu_input_scale", 8.0)),
            precise_relu_target_alphas=tuple(getattr(
                cfg, "precise_relu_target_alphas",
                (7,) if alpha7_backbone else ())),
            precise_relu_lower_degrees=tuple(getattr(
                cfg, "precise_relu_lower_degrees",
                () if alpha7_backbone else (16, 8, 4))),
            precise_relu_progress=float(getattr(
                cfg, "precise_relu_initial_progress", 0.0)),
            precise_relu_backward_mode=str(getattr(
                cfg, "precise_relu_backward_mode", "exact")),
        )
    if cfg.network.startswith("poolformer_no_ln_x2_act"):
        gate_group_epochs = tuple(getattr(
            cfg, "simple_gate_group_epochs", ()))
        model_kwargs.update(
            gate_range_limit=float(getattr(cfg, "simple_gate_range_limit", 6.0)),
            gate_stats_sample_size=int(getattr(
                cfg, "simple_gate_stats_sample_size", 16384)),
            gate_compute_fp32=bool(getattr(
                cfg, "simple_gate_compute_fp32", True)),
            gate_fail_on_nonfinite=bool(getattr(
                cfg, "simple_gate_fail_on_nonfinite", True)),
            gate_initial_blend=float(getattr(
                cfg, "simple_gate_initial_blend",
                0.0 if gate_group_epochs else 1.0)),
            gate_grouping=str(getattr(
                cfg, "simple_gate_grouping", "stage_chunks")),
        )
    if cfg.network.startswith("poolformer_fully_gated_prepbn"):
        model_kwargs.update(
            repbn_bn_eps=float(getattr(cfg, "repbn_bn_eps", 1e-5)),
            repbn_bn_momentum=float(getattr(
                cfg, "repbn_bn_momentum", 0.1)),
            repbn_eta_init=float(getattr(cfg, "repbn_eta_init", 0.0)),
        )
    if cfg.network.startswith("poolformer_fully_gated_affine"):
        model_kwargs.update(
            affine_blocks_per_group=int(getattr(
                cfg, "affine_blocks_per_group", 1)),
        )
    if cfg.network.startswith((
            "poolformer_fully_gated_frozen_std",
            "poolformer_fully_gated_spatial_frozen_std")):
        model_kwargs.update(
            frozen_std_momentum=float(getattr(
                cfg, "frozen_std_momentum", 0.9)),
            frozen_std_initial=float(getattr(
                cfg, "frozen_std_initial", 1.0)),
        )
    if cfg.network.startswith(("poolformer_nf", "iresnet_nf")):
        model_kwargs.update(
            nf_ws_eps=float(getattr(cfg, "nf_ws_eps", 1e-4)),
            nf_tau_init=float(getattr(cfg, "nf_tau_init", 0.1)),
            nf_alpha_init=float(getattr(cfg, "nf_alpha_init", 0.05)),
            nf_alpha_max=float(getattr(cfg, "nf_alpha_max", 0.2)),
            nf_input_gain_init=float(getattr(
                cfg, "nf_input_gain_init", 1.0)),
            nf_input_gain_min=float(getattr(
                cfg, "nf_input_gain_min", 0.25)),
            nf_input_gain_max=float(getattr(
                cfg, "nf_input_gain_max", 4.0)),
            nf_modulator_scale_max=float(getattr(
                cfg, "nf_modulator_scale_max", 0.25)),
            nf_quadratic_scale_max=float(getattr(
                cfg, "nf_quadratic_scale_max",
                getattr(cfg, "nf_modulator_scale_max", 0.25))),
            nf_modulation_input_bound=float(getattr(
                cfg, "nf_modulation_input_bound", 6.0)),
            nf_learnable_ws_gain=bool(getattr(
                cfg, "nf_learnable_ws_gain", True)),
            nf_range_limit=float(getattr(cfg, "nf_range_limit", 6.0)),
            nf_range_sample_size=int(getattr(
                cfg, "nf_range_sample_size", 16384)),
            nf_initial_modulation_progress=float(getattr(
                cfg, "nf_initial_modulation_progress", 1.0)),
            nf_residual_mode=str(getattr(
                cfg, "nf_residual_mode", "convex")),
            nf_fixed_modulator_scale=getattr(
                cfg, "nf_fixed_modulator_scale", None),
        )

    backbone = get_model(cfg.network, **model_kwargs).cuda()
    backbone_init = getattr(cfg, "backbone_init", "")
    if backbone_init and not cfg.resume:
        init_checkpoint = torch.load(backbone_init, map_location="cpu")
        init_metadata = (
            init_checkpoint if isinstance(init_checkpoint, dict) else {})
        init_state = init_metadata.get(
            "state_dict_backbone", init_checkpoint)
        init_loader = getattr(
            backbone, "load_backbone_init_state_dict", None)
        if init_loader is None:
            backbone.load_state_dict(init_state, strict=True)
        else:
            init_loader(init_state)
        serialized_herpn_blends = init_metadata.get("herpn_blends")
        if (isinstance(serialized_herpn_blends, dict)
                and hasattr(backbone, "set_herpn_blends")):
            backbone.set_herpn_blends(serialized_herpn_blends)
            logging.info(
                "Restored %d exact HerPN blends from backbone snapshot",
                len(serialized_herpn_blends))
        elif hasattr(backbone, "set_herpn_progress"):
            backbone.set_herpn_progress(
                float(getattr(
                    cfg, "backbone_init_herpn_progress",
                    getattr(cfg, "herpn_initial_progress", 0.0))))
        if hasattr(backbone, "set_polynomial_progress"):
            backbone.set_polynomial_progress(float(getattr(
                cfg, "precise_relu_initial_progress", 0.0)))
        logging.info("Initialized backbone from %s", backbone_init)
        del init_checkpoint, init_metadata, init_state
    backbone_trainable_prefixes = tuple(str(prefix) for prefix in getattr(
        cfg, "backbone_trainable_prefixes", ()))
    if backbone_trainable_prefixes:
        if any(not prefix for prefix in backbone_trainable_prefixes):
            raise ValueError("backbone_trainable_prefixes cannot contain empty strings")
        trainable_names = []
        for name, parameter in backbone.named_parameters():
            trainable = any(
                name == prefix or name.startswith(prefix + ".")
                for prefix in backbone_trainable_prefixes
            )
            if not trainable:
                parameter.requires_grad_(False)
            elif parameter.requires_grad:
                trainable_names.append(name)
        if not trainable_names:
            raise ValueError(
                "backbone_trainable_prefixes did not select any trainable "
                "backbone parameters")
        logging.info(
            "Restricted backbone training to prefixes %s (%d tensors)",
            backbone_trainable_prefixes, len(trainable_names))
    layerwise_poly_scale_file = getattr(
        cfg, "layerwise_poly_scale_file", "")
    if layerwise_poly_scale_file and not cfg.resume:
        scale_loader = getattr(
            backbone, "load_layerwise_poly_input_scales", None)
        if scale_loader is None:
            raise ValueError(
                "layerwise_poly_scale_file requires a layerwise polynomial "
                "backbone")
        with open(layerwise_poly_scale_file) as scale_handle:
            scale_data = json.load(scale_handle)
        loaded_scale_count = scale_loader(scale_data)
        logging.info(
            "Loaded %d fixed layerwise polynomial input scales from %s",
            loaded_scale_count, layerwise_poly_scale_file)
    affine_calibration_batches = int(getattr(
        cfg, "affine_calibration_batches", 0))
    if affine_calibration_batches > 0 and not cfg.resume:
        completed, diagnostics = calibrate_affine_normalization(
            backbone,
            affine_calibration_loader,
            affine_calibration_batches,
            ridge=float(getattr(cfg, "affine_calibration_ridge", 1e-6)),
            dali=cfg.dali,
        )
        if rank == 0:
            logging.info(
                "Calibrated %d fixed-affine normalization sites with %d "
                "representative batches; worst scale_absmax=%.6g, "
                "bias_absmax=%.6g, relative_rmse_max=%.6g",
                len(diagnostics),
                completed,
                max(item["scale_absmax"] for item in diagnostics),
                max(item["bias_absmax"] for item in diagnostics),
                max(item["relative_rmse_max"] for item in diagnostics),
            )

    embedding_distill_weight = float(getattr(
        cfg, "embedding_distill_weight", 0.0))
    if embedding_distill_weight < 0.0:
        raise ValueError("embedding_distill_weight must be non-negative")
    task_loss_weight = float(getattr(cfg, "task_loss_weight", 1.0))
    if task_loss_weight < 0.0:
        raise ValueError("task_loss_weight must be non-negative")
    embedding_teacher = None
    if embedding_distill_weight > 0.0:
        teacher_network = str(getattr(
            cfg, "embedding_teacher_network", ""))
        teacher_checkpoint = str(getattr(
            cfg, "embedding_teacher_checkpoint", ""))
        if not teacher_network or not teacher_checkpoint:
            raise ValueError(
                "Positive embedding_distill_weight requires both "
                "embedding_teacher_network and embedding_teacher_checkpoint")
        embedding_teacher = get_model(
            teacher_network,
            dropout=0.0,
            fp16=False,
            num_features=cfg.embedding_size,
        ).cuda()
        load_embedding_teacher_checkpoint(
            embedding_teacher, teacher_checkpoint)
        embedding_teacher.eval()
        embedding_teacher.requires_grad_(False)
        logging.info(
            "Loaded frozen embedding teacher %s from %s (weight=%g)",
            teacher_network, teacher_checkpoint, embedding_distill_weight)
    if getattr(cfg, "sync_bn", False):
        backbone = torch.nn.SyncBatchNorm.convert_sync_batchnorm(backbone)

    freeze_batchnorm_running_stats = bool(getattr(
        cfg, "freeze_batchnorm_running_stats", False))
    freeze_batchnorm_affine = bool(getattr(
        cfg, "freeze_batchnorm_affine", False))
    if freeze_batchnorm_affine and not freeze_batchnorm_running_stats:
        raise ValueError(
            "freeze_batchnorm_affine requires "
            "freeze_batchnorm_running_stats")
    if freeze_batchnorm_running_stats:
        frozen_batchnorm_count = freeze_batchnorm_for_training(
            backbone, affine=freeze_batchnorm_affine)
        logging.info(
            "Freezing running statistics for %d BatchNorm modules%s",
            frozen_batchnorm_count,
            " and their affine parameters" if freeze_batchnorm_affine else "",
        )

    backbone = torch.nn.parallel.DistributedDataParallel(
        module=backbone,
        broadcast_buffers=bool(getattr(cfg, "broadcast_buffers", True)),
        device_ids=[local_rank], bucket_cap_mb=16,
        find_unused_parameters=True)
    if getattr(cfg, "ddp_fp16_compress", True):
        backbone.register_comm_hook(None, fp16_compress_hook)

    backbone.train()
    if freeze_batchnorm_running_stats:
        freeze_batchnorm_for_training(
            backbone.module, affine=freeze_batchnorm_affine)
    # FIXME using gradient checkpoint if there are some unused parameters will cause error
    simple_gate_current_group_auxiliary = bool(getattr(
        cfg, "simple_gate_current_group_auxiliary", False))
    if simple_gate_current_group_auxiliary:
        logging.info(
            "Using dynamic DDP graph for current-group-only SimpleGate "
            "auxiliary losses")
    else:
        backbone._set_static_graph()

    cryptoface_patch_training = is_cryptoface_patch_training(cfg)
    if cryptoface_patch_training and world_size != 1:
        raise ValueError("patch_cnn_training='cryptoface' matches the CryptoFace single-GPU training loop; use one process.")

    margin_loss = build_margin_loss(cfg)
    use_layerwise_staged_optimizer = bool(getattr(
        cfg, "layerwise_poly_staged_training", False))
    layerwise_parameter_getter = (
        getattr(backbone.module, "layerwise_poly_parameters", None)
        if use_layerwise_staged_optimizer else None
    )
    layerwise_poly_parameters = (
        list(layerwise_parameter_getter())
        if layerwise_parameter_getter is not None else []
    )
    layerwise_poly_parameter_ids = {
        id(parameter) for parameter in layerwise_poly_parameters
    }

    if cryptoface_patch_training:
        module_partial_fc = CryptoFaceArcFaceHead(
            cfg.embedding_size,
            cfg.num_classes,
            s=float(getattr(cfg, "scale", 64.0)),
            m=float(getattr(cfg, "cryptoface_arcface_margin", cfg.margin_list[1])),
        ).cuda()
        criterion = nn.CrossEntropyLoss()
        opt = torch.optim.SGD(
            params=[
                {
                    "params": [module_partial_fc.kernel],
                    "weight_decay": cfg.weight_decay,
                    "scope": "classifier",
                },
                {
                    "params": [
                        parameter for parameter in backbone.parameters()
                        if (parameter.requires_grad
                            and id(parameter) not in layerwise_poly_parameter_ids)
                    ],
                    "scope": "backbone",
                },
                *([{
                    "params": layerwise_poly_parameters,
                    "weight_decay": 0.0,
                    "scope": "layerwise_poly",
                }] if layerwise_poly_parameters else []),
            ],
            lr=cfg.lr,
            momentum=cfg.momentum,
        )
    elif cfg.optimizer == "sgd":
        module_partial_fc = PartialFC_V2(
            margin_loss, cfg.embedding_size, cfg.num_classes,
            cfg.sample_rate, False)
        module_partial_fc.train().cuda()
        # TODO the params of partial fc must be last in the params list
        if getattr(cfg, "selective_weight_decay", False):
            decay_params, no_decay_params = split_weight_decay_parameters(
                backbone)
            decay_params = [
                parameter for parameter in decay_params
                if id(parameter) not in layerwise_poly_parameter_ids
            ]
            no_decay_params = [
                parameter for parameter in no_decay_params
                if id(parameter) not in layerwise_poly_parameter_ids
            ]
            parameter_groups = [
                {
                    "params": decay_params,
                    "weight_decay": cfg.weight_decay,
                    "scope": "backbone",
                },
                {
                    "params": no_decay_params,
                    "weight_decay": 0.0,
                    "scope": "backbone",
                },
                *([{
                    "params": layerwise_poly_parameters,
                    "weight_decay": 0.0,
                    "scope": "layerwise_poly",
                }] if layerwise_poly_parameters else []),
                {
                    "params": module_partial_fc.parameters(),
                    "weight_decay": cfg.weight_decay,
                    "scope": "classifier",
                },
            ]
            logging.info(
                "Selective weight decay: %d backbone tensors with decay, "
                "%d without decay",
                len(decay_params), len(no_decay_params))
        else:
            parameter_groups = [
                {
                    "params": [
                        parameter for parameter in backbone.parameters()
                        if (parameter.requires_grad
                            and id(parameter) not in layerwise_poly_parameter_ids)
                    ],
                    "scope": "backbone",
                },
                *([{
                    "params": layerwise_poly_parameters,
                    "weight_decay": 0.0,
                    "scope": "layerwise_poly",
                }] if layerwise_poly_parameters else []),
                {
                    "params": module_partial_fc.parameters(),
                    "scope": "classifier",
                },
            ]
        opt = torch.optim.SGD(
            params=parameter_groups,
            lr=cfg.lr,
            momentum=cfg.momentum,
            weight_decay=(0.0 if getattr(
                cfg, "selective_weight_decay", False)
                else cfg.weight_decay))

    elif cfg.optimizer == "adamw":
        module_partial_fc = PartialFC_V2(
            margin_loss, cfg.embedding_size, cfg.num_classes,
            cfg.sample_rate, False)
        module_partial_fc.train().cuda()
        if getattr(cfg, "selective_weight_decay", False):
            decay_params, no_decay_params = split_weight_decay_parameters(
                backbone)
            parameter_groups = [
                {
                    "params": decay_params,
                    "weight_decay": cfg.weight_decay,
                    "scope": "backbone",
                },
                {
                    "params": no_decay_params,
                    "weight_decay": 0.0,
                    "scope": "backbone",
                },
                {
                    "params": module_partial_fc.parameters(),
                    "weight_decay": cfg.weight_decay,
                    "scope": "classifier",
                },
            ]
            logging.info(
                "Selective AdamW weight decay: %d backbone tensors with "
                "decay, %d without decay",
                len(decay_params), len(no_decay_params))
        else:
            parameter_groups = [
                {
                    "params": [
                        parameter for parameter in backbone.parameters()
                        if (parameter.requires_grad
                            and id(parameter) not in
                            layerwise_poly_parameter_ids)
                    ],
                    "scope": "backbone",
                },
                *([{
                    "params": layerwise_poly_parameters,
                    "weight_decay": 0.0,
                    "scope": "layerwise_poly",
                }] if layerwise_poly_parameters else []),
                {
                    "params": module_partial_fc.parameters(),
                    "scope": "classifier",
                },
            ]
        opt = torch.optim.AdamW(
            params=parameter_groups,
            lr=cfg.lr,
            weight_decay=(0.0 if getattr(
                cfg, "selective_weight_decay", False)
                else cfg.weight_decay))
    else:
        raise

    cfg.total_batch_size = cfg.batch_size * world_size
    cfg.warmup_step = cfg.num_image // cfg.total_batch_size * cfg.warmup_epoch
    cfg.total_step = cfg.num_image // cfg.total_batch_size * cfg.num_epoch
    steps_per_epoch = cfg.num_image // cfg.total_batch_size
    prepbn_decay_epochs = getattr(cfg, "prepbn_decay_epochs", None)
    if prepbn_decay_epochs is not None:
        prepbn_decay_steps = int(steps_per_epoch * prepbn_decay_epochs)
    else:
        prepbn_decay_steps = int(getattr(cfg, "prepbn_decay_steps", cfg.total_step))
    if (getattr(cfg, "prepbn_require_full_transition", False)
            and prepbn_decay_steps > cfg.total_step):
        raise ValueError(
            "RepBatchNorm transition does not finish before training ends: "
            f"decay_steps={prepbn_decay_steps}, total_steps={cfg.total_step}")

    lr_scheduler_name = str(getattr(cfg, "lr_scheduler", "polynomial"))
    lr_scheduler_step_per_epoch = lr_scheduler_name == "multistep"
    if lr_scheduler_step_per_epoch:
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            opt,
            milestones=list(getattr(cfg, "lr_milestones", [12, 20, 24])),
            gamma=float(getattr(cfg, "lr_gamma", 0.1)),
        )
    elif lr_scheduler_name == "polynomial":
        lr_scheduler = PolynomialLRWarmup(
            optimizer=opt,
            warmup_iters=cfg.warmup_step,
            total_iters=cfg.total_step)
    elif lr_scheduler_name == "cosine":
        lr_scheduler = CosineLRWarmup(
            optimizer=opt,
            warmup_iters=cfg.warmup_step,
            total_iters=cfg.total_step,
            min_lr_ratio=float(getattr(cfg, "min_lr_ratio", 0.01)),
        )
    else:
        raise ValueError(
            f"Unknown lr_scheduler {lr_scheduler_name!r}; expected "
            "'polynomial', 'cosine', or 'multistep'")

    start_epoch = 0
    global_step = 0
    resumed_completed_simple_gate_groups = None
    resumed_repbn_gate_recalibrated = None
    resumed_affine_group_epochs = None
    resumed_affine_group_names = None
    resumed_completed_herpn_groups = None
    resumed_herpn_conversion_groups = None
    resumed_frozen_std_group_names = None
    resumed_frozen_std_group_steps = None
    resumed_frozen_std_transition_steps = None
    resumed_completed_frozen_std_groups = None
    if cfg.resume:
        resume_checkpoint_dir = str(getattr(
            cfg, "resume_checkpoint_dir", cfg.output))
        resume_checkpoint_path = os.path.join(
            resume_checkpoint_dir, f"checkpoint_gpu_{rank}.pt")
        dict_checkpoint = torch.load(resume_checkpoint_path)
        if rank == 0:
            logging.info(
                "Resuming distributed training state from %s; new "
                "checkpoints will be written to %s",
                resume_checkpoint_dir, cfg.output)
        start_epoch = dict_checkpoint["epoch"]
        global_step = dict_checkpoint["global_step"]
        resumed_completed_simple_gate_groups = dict_checkpoint.get(
            "completed_simple_gate_groups")
        resumed_repbn_gate_recalibrated = dict_checkpoint.get(
            "repbn_gate_recalibrated")
        resumed_affine_group_epochs = dict_checkpoint.get(
            "affine_group_epochs")
        resumed_affine_group_names = dict_checkpoint.get(
            "affine_group_names")
        resumed_completed_herpn_groups = dict_checkpoint.get(
            "completed_herpn_groups")
        resumed_herpn_conversion_groups = dict_checkpoint.get(
            "herpn_conversion_groups")
        resumed_frozen_std_group_names = dict_checkpoint.get(
            "frozen_std_group_names")
        resumed_frozen_std_group_steps = dict_checkpoint.get(
            "frozen_std_group_steps")
        resumed_frozen_std_transition_steps = dict_checkpoint.get(
            "frozen_std_transition_steps")
        resumed_completed_frozen_std_groups = dict_checkpoint.get(
            "completed_frozen_std_groups")
        checkpoint_gate_grouping = dict_checkpoint.get(
            "simple_gate_grouping")
        configured_gate_grouping = str(getattr(
            cfg, "simple_gate_grouping", "stage_chunks"))
        if (checkpoint_gate_grouping is not None
                and checkpoint_gate_grouping != configured_gate_grouping):
            raise ValueError(
                "Resume checkpoint SimpleGate grouping "
                f"{checkpoint_gate_grouping!r} does not match config "
                f"{configured_gate_grouping!r}")
        backbone.module.load_state_dict(dict_checkpoint["state_dict_backbone"])
        if hasattr(backbone.module, "set_pillar_input_scales"):
            backbone.module.set_pillar_input_scales(
                dict(getattr(cfg, "pillar_input_scale_overrides", {})),
                default_input_scale=float(getattr(
                    cfg, "pillar_input_scale", 1.0)),
            )
        module_partial_fc.load_state_dict(dict_checkpoint["state_dict_softmax_fc"])
        resume_optimizer_state = bool(getattr(
            cfg, "resume_optimizer_state", True))
        if resume_optimizer_state:
            opt.load_state_dict(dict_checkpoint["state_optimizer"])
        elif rank == 0:
            logging.warning(
                "Resume is using a freshly initialized optimizer because "
                "resume_optimizer_state=False")
        if getattr(cfg, "resume_rebase_lr_scheduler", False):
            # An accuracy-recovery run can deliberately extend ``num_epoch``
            # beyond the checkpoint's original schedule.  Loading that
            # exhausted scheduler would leave every optimizer group at zero
            # LR.  Rebase the freshly constructed scheduler at the resumed
            # global step so it uses the new total-step horizon and restores
            # a non-zero closed-form LR from its configured base rates.
            lr_scheduler.step(global_step)
            if rank == 0:
                logging.info(
                    "Rebased LR scheduler at resumed global_step=%d: "
                    "lr=%s total_step=%d",
                    global_step,
                    [group["lr"] for group in opt.param_groups],
                    cfg.total_step)
        elif resume_optimizer_state:
            lr_scheduler.load_state_dict(dict_checkpoint["state_lr_scheduler"])
        else:
            raise ValueError(
                "resume_optimizer_state=False requires "
                "resume_rebase_lr_scheduler=True so the fresh optimizer "
                "does not restart at the initial learning rate")
        legacy_parameters = getattr(
            backbone.module, "legacy_layerwise_poly_parameters", lambda: [])()
        for parameter in legacy_parameters:
            opt.state.pop(parameter, None)
        if legacy_parameters and rank == 0:
            logging.warning(
                "Migrated %d legacy layerwise theta2 parameters to stable "
                "beta2 coordinates; discarded their incompatible SGD state",
                len(legacy_parameters))
        del dict_checkpoint

    for key, value in cfg.items():
        num_space = 25 - len(key)
        logging.info(": " + key + " " * num_space + str(value))

    callback_verification = CallBackVerification(
        val_targets=cfg.val_targets, rec_prefix=cfg.rec, 
        summary_writer=summary_writer, wandb_logger=wandb_logger,
        fail_on_nonfinite=getattr(cfg, "fail_on_nonfinite_val", False),
        max_embedding_abs=getattr(cfg, "max_validation_embedding_abs", None),
        batch_size=getattr(cfg, "validation_batch_size", 10),
    )
    callback_logging = CallBackLogging(
        frequent=cfg.frequent,
        total_step=cfg.total_step,
        batch_size=cfg.batch_size,
        start_step = global_step,
        writer=summary_writer
    )

    loss_am = AverageMeter()
    amp = torch.cuda.amp.grad_scaler.GradScaler(
        init_scale=float(getattr(cfg, "amp_init_scale", 65536.0)),
        growth_interval=int(getattr(cfg, "amp_growth_interval", 100)),
    )
    skip_nonfinite_gradients = bool(getattr(
        cfg, "skip_nonfinite_gradients", cfg.fp16))
    nonfinite_gradient_scale_backoff = float(getattr(
        cfg, "nonfinite_gradient_scale_backoff", 0.5))
    min_amp_scale = float(getattr(cfg, "min_amp_scale", 1.0))
    max_nonfinite_gradient_skips = int(getattr(
        cfg, "max_nonfinite_gradient_skips", 0))
    if not 0.0 < nonfinite_gradient_scale_backoff < 1.0:
        raise ValueError(
            "nonfinite_gradient_scale_backoff must be in (0, 1)")
    if min_amp_scale <= 0.0:
        raise ValueError("min_amp_scale must be positive")
    if max_nonfinite_gradient_skips < 0:
        raise ValueError(
            "max_nonfinite_gradient_skips must be non-negative")
    nonfinite_gradient_skips = 0
    if rank == 0 and cfg.fp16:
        logging.info(
            "FP16 non-finite gradient policy: %s, scale_backoff=%.4g, "
            "minimum_scale=%.4g, maximum_skips=%s",
            "warn and skip" if skip_nonfinite_gradients else "fail fast",
            nonfinite_gradient_scale_backoff,
            min_amp_scale,
            (str(max_nonfinite_gradient_skips)
             if max_nonfinite_gradient_skips > 0 else "unlimited"),
        )
    grad_clip = float(getattr(cfg, "gradient_clip", 5.0))
    gradient_clip_scope = str(getattr(cfg, "gradient_clip_scope", "all"))
    clipped_params = select_gradient_clip_parameters(
        opt, backbone, scope=gradient_clip_scope)
    if rank == 0:
        logging.info(
            "Gradient clipping: scope=%s, max=%g, tensors=%d",
            gradient_clip_scope, grad_clip, len(clipped_params))

    herpn_stage_epochs = tuple(getattr(cfg, "herpn_stage_epochs", ()))
    herpn_conversion_groups = tuple(
        tuple(group) for group in getattr(cfg, "herpn_conversion_groups", ()))
    herpn_group_epochs = tuple(getattr(cfg, "herpn_group_epochs", ()))
    herpn_transition_epochs = float(getattr(cfg, "herpn_transition_epochs", 1.0))
    herpn_range_loss_weight = float(getattr(cfg, "herpn_range_loss_weight", 0.0))
    herpn_distill_loss_weight = float(getattr(cfg, "herpn_distill_loss_weight", 0.0))
    herpn_range_loss_names = tuple(str(name) for name in getattr(
        cfg, "herpn_range_loss_names", ()))
    herpn_bn_recalibration_batches = int(
        getattr(cfg, "herpn_bn_recalibration_batches", 0))
    herpn_save_after_group = bool(
        getattr(cfg, "herpn_save_after_group", False))
    pillar_enabled = hasattr(backbone.module, "pillar_range_penalty")
    # PILLAR reuses the activation-factory iResNet class, whose dormant HerPN
    # methods are implementation details rather than a conversion curriculum.
    herpn_enabled = (
        hasattr(backbone.module, "set_herpn_progress")
        and not pillar_enabled)
    pillar_target_coefficient = float(getattr(
        cfg, "pillar_regularization_coefficient", 0.0))
    pillar_target_exponent = int(getattr(
        cfg, "pillar_regularization_exponent", 10))
    pillar_regularization_warmup = bool(getattr(
        cfg, "pillar_regularization_warmup", True))
    pillar_range_only_epochs = int(getattr(
        cfg, "pillar_range_only_epochs", 0))
    pillar_task_loss_weight = 1.0
    pillar_log_interval = int(getattr(cfg, "pillar_log_interval", 0))
    pillar_skip_verification_epochs = int(getattr(
        cfg, "pillar_skip_verification_epochs", 0))
    pillar_strict_verification_epoch = int(getattr(
        cfg, "pillar_strict_verification_epoch",
        pillar_skip_verification_epochs))
    pillar_effective_coefficient = pillar_target_coefficient
    pillar_effective_exponent = pillar_target_exponent
    if pillar_target_coefficient < 0.0:
        raise ValueError(
            "pillar_regularization_coefficient must be non-negative")
    if pillar_target_exponent < 4 or pillar_target_exponent % 2:
        raise ValueError(
            "pillar_regularization_exponent must be an even integer >= 4")
    if pillar_range_only_epochs < 0:
        raise ValueError("pillar_range_only_epochs must be non-negative")
    if pillar_log_interval < 0:
        raise ValueError("pillar_log_interval must be non-negative")
    if pillar_skip_verification_epochs < 0:
        raise ValueError(
            "pillar_skip_verification_epochs must be non-negative")
    if pillar_strict_verification_epoch < pillar_skip_verification_epochs:
        raise ValueError(
            "pillar_strict_verification_epoch must be no smaller than "
            "pillar_skip_verification_epochs")
    if pillar_target_coefficient > 0.0 and not pillar_enabled:
        raise ValueError(
            "pillar_regularization_coefficient requires a PILLAR backbone")
    if pillar_enabled:
        pillar_effective_coefficient, pillar_effective_exponent = (
            pillar_regularization_at_epoch(
                start_epoch,
                pillar_target_coefficient,
                pillar_target_exponent,
                warmup=pillar_regularization_warmup,
            )
        )
        backbone.module.set_pillar_regularization_exponent(
            pillar_effective_exponent)
        pillar_task_loss_weight = pillar_task_loss_weight_at_epoch(
            start_epoch, pillar_range_only_epochs)
        if rank == 0:
            logging.info(
                "PILLAR regularization: target_beta=%g target_gamma=%d "
                "warmup=%s current_beta=%g current_gamma=%d "
                "range_only_epochs=%d task_loss_weight=%g",
                pillar_target_coefficient, pillar_target_exponent,
                pillar_regularization_warmup,
                pillar_effective_coefficient, pillar_effective_exponent,
                pillar_range_only_epochs, pillar_task_loss_weight,
            )
    precise_relu_enabled = hasattr(
        backbone.module, "set_polynomial_progress")
    precise_relu_stage_epochs = tuple(getattr(
        cfg, "precise_relu_stage_epochs", ()))
    precise_relu_transition_epochs = float(getattr(
        cfg, "precise_relu_transition_epochs", 1.0))
    precise_relu_range_loss_weight = float(getattr(
        cfg, "precise_relu_range_loss_weight", 0.0))
    precise_relu_bn_recalibration_batches = int(getattr(
        cfg, "precise_relu_bn_recalibration_batches", 0))
    nf_enabled = hasattr(backbone.module, "set_nf_range_tracking")
    nf_range_loss_weight = float(getattr(
        cfg, "nf_range_loss_weight", 0.0))
    nf_stats_interval = int(getattr(cfg, "nf_stats_interval", 0))
    nf_modulation_group_epochs = tuple(getattr(
        cfg, "nf_modulation_group_epochs", ()))
    nf_modulation_transition_epochs = float(getattr(
        cfg, "nf_modulation_transition_epochs", 1.0))
    nf_modulation_order = str(getattr(
        cfg, "nf_modulation_order", "reverse"))
    nf_modulation_setter = getattr(
        backbone.module, "set_nf_modulation_progresses", None)
    nf_modulation_schedule = bool(nf_modulation_group_epochs)
    if nf_range_loss_weight < 0.0:
        raise ValueError("nf_range_loss_weight must be non-negative")
    if nf_stats_interval < 0:
        raise ValueError("nf_stats_interval must be non-negative")
    if nf_range_loss_weight > 0.0 and not nf_enabled:
        raise ValueError(
            "nf_range_loss_weight requires a normalization-free backbone")
    if nf_modulation_schedule:
        if nf_modulation_setter is None:
            raise ValueError(
                "nf_modulation_group_epochs requires a backbone with "
                "set_nf_modulation_progresses")
        group_count = int(backbone.module.nf_modulation_group_count())
        if len(nf_modulation_group_epochs) != group_count:
            raise ValueError(
                "nf_modulation_group_epochs must contain one start per "
                f"quadratic block ({group_count}), got "
                f"{len(nf_modulation_group_epochs)}")
        final_progresses = simple_gate_blends_at_epoch(
            cfg.num_epoch,
            nf_modulation_group_epochs,
            nf_modulation_transition_epochs,
        )
        if (getattr(cfg, "nf_require_full_modulation", True)
                and min(final_progresses, default=0.0) < 1.0):
            raise ValueError(
                "NF modulation schedule does not finish before training ends")
        nf_modulation_setter(
            simple_gate_blends_at_epoch(
                start_epoch,
                nf_modulation_group_epochs,
                nf_modulation_transition_epochs,
            ),
            order=nf_modulation_order,
        )
        if rank == 0:
            logging.info(
                "NF quadratic schedule: groups=%d order=%s starts=%s "
                "transition_epochs=%g",
                group_count, nf_modulation_order,
                nf_modulation_group_epochs,
                nf_modulation_transition_epochs,
            )
    herpn_group_schedule = bool(herpn_enabled and herpn_conversion_groups)
    precise_relu_schedule = bool(
        precise_relu_enabled and precise_relu_stage_epochs)
    if precise_relu_range_loss_weight < 0.0:
        raise ValueError("precise_relu_range_loss_weight must be non-negative")
    if precise_relu_range_loss_weight > 0.0 and not precise_relu_enabled:
        raise ValueError(
            "precise_relu_range_loss_weight requires a precise-ReLU backbone")
    if precise_relu_bn_recalibration_batches < 0:
        raise ValueError(
            "precise_relu_bn_recalibration_batches must be non-negative")
    if precise_relu_schedule:
        transition_count = int(
            backbone.module.polynomial_transition_count())
        if len(precise_relu_stage_epochs) != transition_count:
            raise ValueError(
                "precise_relu_stage_epochs must contain one start for each "
                f"polynomial student ({transition_count}), got "
                f"{len(precise_relu_stage_epochs)}")
        final_progress = precise_relu_progress_at_epoch(
            cfg.num_epoch,
            precise_relu_stage_epochs,
            precise_relu_transition_epochs,
        )
        require_final_stage = bool(getattr(
            cfg, "precise_relu_require_final_stage",
            getattr(cfg, "precise_relu_require_final_degree", True)))
        if (require_final_stage
                and final_progress < transition_count):
            raise ValueError(
                "PreciseReLU schedule does not reach its final stage before "
                f"training ends: final_progress={final_progress:.3f}")
        backbone.module.set_polynomial_progress(
            precise_relu_progress_at_epoch(
                start_epoch,
                precise_relu_stage_epochs,
                precise_relu_transition_epochs,
            ))
        if rank == 0:
            logging.info(
                "PreciseReLU curriculum: stages=%s starts=%s "
                "transition_epochs=%g input_scale=%g backward=%s",
                backbone.module.polynomial_stage_names(),
                precise_relu_stage_epochs,
                precise_relu_transition_epochs,
                float(getattr(cfg, "precise_relu_input_scale", 8.0)),
                str(getattr(cfg, "precise_relu_backward_mode", "exact")),
            )
    layerwise_poly_enabled = (
        hasattr(backbone.module, "layerwise_poly_activation_names")
        and bool(getattr(
            backbone.module, "layerwise_input_scale_enabled", True)))
    layerwise_poly_range_batches = int(getattr(
        cfg, "layerwise_poly_range_calibration_batches", 0))
    layerwise_poly_range_margin = float(getattr(
        cfg, "layerwise_poly_range_margin", 1.1))
    layerwise_poly_min_scale = float(getattr(
        cfg, "layerwise_poly_min_scale", 1e-3))
    layerwise_poly_range_quantile = float(getattr(
        cfg, "layerwise_poly_range_quantile", 1.0))
    layerwise_poly_quantile_samples = int(getattr(
        cfg, "layerwise_poly_quantile_samples", 65536))
    layerwise_poly_range_holdout_fraction = float(getattr(
        cfg, "layerwise_poly_range_holdout_fraction", 0.0))
    layerwise_poly_max_tail_ratio = float(getattr(
        cfg, "layerwise_poly_max_tail_ratio", 0.0))
    layerwise_poly_max_scale_growth = float(getattr(
        cfg, "layerwise_poly_max_scale_growth", 0.0))
    layerwise_poly_max_input_scale = float(getattr(
        cfg, "layerwise_poly_max_input_scale", 0.0))
    layerwise_poly_require_full_containment = bool(getattr(
        cfg, "layerwise_poly_require_full_containment", False))
    layerwise_poly_scan_both_orientations = bool(getattr(
        cfg, "layerwise_poly_scan_both_orientations", False))
    layerwise_poly_freeze_containment_interval = bool(getattr(
        cfg, "layerwise_poly_freeze_containment_interval", False))
    layerwise_poly_tail_topk = int(getattr(
        cfg, "layerwise_poly_tail_topk", 0))
    layerwise_poly_tail_replay_batch_size = int(getattr(
        cfg, "layerwise_poly_tail_replay_batch_size", 0))
    layerwise_poly_tail_replay_workers = int(getattr(
        cfg, "layerwise_poly_tail_replay_workers", 0))
    layerwise_poly_tail_replay_priority_count = int(getattr(
        cfg, "layerwise_poly_tail_replay_priority_count", 0))
    layerwise_poly_tail_replay_priority_repeats = int(getattr(
        cfg, "layerwise_poly_tail_replay_priority_repeats", 1))
    layerwise_poly_tail_replay_extra_indices = tuple(getattr(
        cfg, "layerwise_poly_tail_replay_extra_indices", ()))
    layerwise_poly_staged_training = bool(getattr(
        cfg, "layerwise_poly_staged_training", False))
    layerwise_poly_freeze_backbone_during_local_fit = bool(getattr(
        cfg, "layerwise_poly_freeze_backbone_during_local_fit", False))
    layerwise_poly_preserve_batchnorm_during_local_fit = bool(getattr(
        cfg, "layerwise_poly_preserve_batchnorm_during_local_fit", False))
    layerwise_poly_preserve_batchnorm_during_blend = bool(getattr(
        cfg, "layerwise_poly_preserve_batchnorm_during_blend", False))
    layerwise_poly_preserve_batchnorm_during_final_finetune = bool(getattr(
        cfg, "layerwise_poly_preserve_batchnorm_during_final_finetune", False))
    layerwise_poly_conditioning_backbone_lr_scale = float(getattr(
        cfg, "layerwise_poly_conditioning_backbone_lr_scale", 0.0))
    layerwise_poly_conditioning_range_loss_weight = float(getattr(
        cfg, "layerwise_poly_conditioning_range_loss_weight",
        getattr(cfg, "herpn_range_loss_weight", 0.0)))
    layerwise_poly_allow_provisional_tail = bool(getattr(
        cfg, "layerwise_poly_allow_provisional_tail_conditioning", False))
    layerwise_poly_initial_calibration_provisional = bool(getattr(
        cfg, "layerwise_poly_initial_calibration_provisional", False))
    layerwise_poly_strict_recalibrate_before_blend = bool(getattr(
        cfg, "layerwise_poly_strict_recalibrate_before_blend", False))
    layerwise_poly_causal_strict_calibration = bool(getattr(
        cfg, "layerwise_poly_causal_strict_calibration", False))
    layerwise_poly_verify_singleton_boundary = bool(getattr(
        cfg, "layerwise_poly_verify_singleton_boundary", False))
    layerwise_poly_calibration_log_interval = int(getattr(
        cfg, "layerwise_poly_calibration_log_interval", 0))
    layerwise_poly_strict_tail_scale_floor = bool(getattr(
        cfg, "layerwise_poly_strict_tail_scale_floor", False))
    layerwise_poly_tail_scale_floor_margin = float(getattr(
        cfg, "layerwise_poly_tail_scale_floor_margin", 1.0))
    layerwise_poly_max_tail_scale_expansion = float(getattr(
        cfg, "layerwise_poly_max_tail_scale_expansion", 0.0))
    layerwise_poly_blend_backbone_lr_scale = float(getattr(
        cfg, "layerwise_poly_blend_backbone_lr_scale", 1.0))
    layerwise_poly_final_backbone_lr_scale = float(getattr(
        cfg, "layerwise_poly_final_backbone_lr_scale", 1.0))
    layerwise_poly_optimizer_lr_scale = float(getattr(
        cfg, "layerwise_poly_optimizer_lr_scale", 1.0))
    layerwise_poly_allow_selective_order = bool(getattr(
        cfg, "layerwise_poly_allow_selective_order", False))
    if min(
            layerwise_poly_blend_backbone_lr_scale,
            layerwise_poly_final_backbone_lr_scale,
            layerwise_poly_optimizer_lr_scale) <= 0.0:
        raise ValueError(
            "Layerwise polynomial optimizer LR scales must be positive")
    if layerwise_poly_conditioning_backbone_lr_scale < 0.0:
        raise ValueError(
            "layerwise_poly_conditioning_backbone_lr_scale must be non-negative")
    if layerwise_poly_conditioning_range_loss_weight < 0.0:
        raise ValueError(
            "layerwise_poly_conditioning_range_loss_weight must be non-negative")
    if layerwise_poly_calibration_log_interval < 0:
        raise ValueError(
            "layerwise_poly_calibration_log_interval must be non-negative")
    if any(
            type(index) is not int or index < 0
            for index in layerwise_poly_tail_replay_extra_indices):
        raise ValueError(
            "layerwise_poly_tail_replay_extra_indices must contain "
            "non-negative integers")
    if layerwise_poly_tail_topk < 0:
        raise ValueError("layerwise_poly_tail_topk must be non-negative")
    if min(
            layerwise_poly_tail_replay_batch_size,
            layerwise_poly_tail_replay_workers) < 0:
        raise ValueError(
            "Layerwise tail replay batch size/workers must be non-negative")
    if layerwise_poly_tail_replay_priority_count < 0:
        raise ValueError(
            "Layerwise tail replay priority count must be non-negative")
    if layerwise_poly_tail_replay_priority_repeats < 1:
        raise ValueError(
            "Layerwise tail replay priority repeats must be positive")
    if (layerwise_poly_tail_replay_priority_count > 0
            and layerwise_poly_tail_replay_batch_size <= 0):
        raise ValueError(
            "Prioritized layerwise tail replay requires a positive replay "
            "batch size")
    if (layerwise_poly_tail_replay_batch_size > 0
            and layerwise_poly_tail_topk <= 0):
        raise ValueError(
            "Layerwise tail replay requires layerwise_poly_tail_topk > 0")
    if layerwise_poly_tail_topk > 0 and cfg.dali:
        raise ValueError(
            "Layerwise tail sample indexing requires config.dali=False")
    if (layerwise_poly_freeze_containment_interval
            and not layerwise_poly_require_full_containment):
        raise ValueError(
            "A frozen containment interval requires full containment")
    if (layerwise_poly_require_full_containment
            and (not layerwise_poly_allow_provisional_tail
                 or not layerwise_poly_strict_recalibrate_before_blend)):
        raise ValueError(
            "Full containment requires provisional conditioning and strict "
            "pre-blend verification")
    if (layerwise_poly_causal_strict_calibration
            and not layerwise_poly_strict_recalibrate_before_blend):
        raise ValueError(
            "Causal layerwise calibration requires strict pre-blend "
            "recalibration")
    if ((layerwise_poly_preserve_batchnorm_during_local_fit
         or layerwise_poly_preserve_batchnorm_during_blend
         or layerwise_poly_preserve_batchnorm_during_final_finetune)
            and not layerwise_poly_staged_training):
        raise ValueError(
            "Phase-specific BatchNorm preservation requires staged layerwise "
            "polynomial training")
    if (layerwise_poly_verify_singleton_boundary
            and not layerwise_poly_causal_strict_calibration):
        raise ValueError(
            "Singleton boundary verification requires causal strict "
            "calibration")
    if layerwise_poly_tail_scale_floor_margin < 1.0:
        raise ValueError(
            "layerwise_poly_tail_scale_floor_margin must be at least 1")
    if (layerwise_poly_max_tail_scale_expansion < 0.0
            or 0.0 < layerwise_poly_max_tail_scale_expansion < 1.0):
        raise ValueError(
            "layerwise_poly_max_tail_scale_expansion must be zero or at least 1")
    if (layerwise_poly_strict_tail_scale_floor
            and (not layerwise_poly_strict_recalibrate_before_blend
                 or layerwise_poly_max_tail_ratio <= 0.0)):
        raise ValueError(
            "Strict tail scale flooring requires strict pre-blend "
            "recalibration and a positive maximum tail ratio")
    if (layerwise_poly_allow_provisional_tail
            and (not layerwise_poly_staged_training
                 or layerwise_poly_conditioning_backbone_lr_scale <= 0.0
                 or layerwise_poly_conditioning_range_loss_weight <= 0.0
                 or not layerwise_poly_strict_recalibrate_before_blend)):
        raise ValueError(
            "Provisional layerwise intervals require staged training, a "
            "positive conditioning backbone LR/range weight, and strict "
            "recalibration before blending")
    if (layerwise_poly_initial_calibration_provisional
            and not layerwise_poly_allow_provisional_tail):
        raise ValueError(
            "Provisional initial layerwise calibration requires provisional "
            "tail conditioning")
    layerwise_poly_local_fit_backbone_lr_scale = (
        layerwise_poly_conditioning_backbone_lr_scale
        if layerwise_poly_conditioning_backbone_lr_scale > 0.0
        else 1.0
    )
    if herpn_group_schedule:
        validate_herpn_conversion_groups(
            backbone.module, herpn_conversion_groups)
        if (layerwise_poly_require_full_containment
                and any(len(group) != 1 for group in herpn_conversion_groups)):
            raise ValueError(
                "Full-containment training requires singleton conversion "
                "groups so every polynomial boundary is audited separately")
        if resumed_herpn_conversion_groups is not None:
            checkpoint_groups = tuple(
                tuple(group) for group in resumed_herpn_conversion_groups)
            if checkpoint_groups != herpn_conversion_groups:
                raise ValueError(
                    "Resume checkpoint HerPN conversion groups do not match "
                    f"the config: checkpoint={checkpoint_groups}, "
                    f"config={herpn_conversion_groups}")
        final_blends = herpn_group_blends_at_epoch(
            cfg.num_epoch, herpn_conversion_groups, herpn_group_epochs,
            herpn_transition_epochs)
        if (getattr(cfg, "herpn_require_full_conversion", True)
                and min(final_blends.values(), default=0.0) < 1.0):
            raise ValueError(
                "HerPN group schedule does not finish before training ends")
    elif herpn_enabled and herpn_stage_epochs:
        final_progress = herpn_progress_at_epoch(
            cfg.num_epoch, herpn_stage_epochs, herpn_transition_epochs)
        if getattr(cfg, "herpn_require_full_conversion", True) and final_progress < 5.0:
            raise ValueError(
                "HerPN schedule does not finish all five stages before training ends: "
                f"final_progress={final_progress:.3f}"
            )
    if layerwise_poly_enabled:
        if not herpn_group_schedule:
            raise ValueError(
                "Layerwise polynomial training requires herpn_conversion_groups")
        expected_order = tuple(
            backbone.module.layerwise_poly_activation_names())
        configured_order = tuple(
            name for group in herpn_conversion_groups for name in group)
        if (configured_order != expected_order
                and not layerwise_poly_allow_selective_order):
            raise ValueError(
                "Layerwise polynomial activations must convert in forward order; "
                f"configured={configured_order}, expected={expected_order}")
        if layerwise_poly_range_batches < 0:
            raise ValueError(
                "layerwise_poly_range_calibration_batches must be >= 0")
        if layerwise_poly_range_margin < 1.0:
            raise ValueError("layerwise_poly_range_margin must be at least 1")
        if layerwise_poly_min_scale <= 0.0:
            raise ValueError("layerwise_poly_min_scale must be positive")
        if not 0.0 < layerwise_poly_range_quantile <= 1.0:
            raise ValueError(
                "layerwise_poly_range_quantile must be in (0, 1]")
        if layerwise_poly_quantile_samples <= 0:
            raise ValueError(
                "layerwise_poly_quantile_samples must be positive")
        if not 0.0 <= layerwise_poly_range_holdout_fraction < 0.5:
            raise ValueError(
                "layerwise_poly_range_holdout_fraction must be in [0, 0.5)")
        if min(
                layerwise_poly_max_tail_ratio,
                layerwise_poly_max_scale_growth,
                layerwise_poly_max_input_scale) < 0.0:
            raise ValueError(
                "Layerwise polynomial safety limits must be non-negative")
        fractional_group_starts = tuple(
            start for start in herpn_group_epochs
            if abs(float(start) - round(float(start))) > 1e-9)
        if (fractional_group_starts
                and layerwise_poly_strict_recalibrate_before_blend
                and cfg.dali):
            raise ValueError(
                "Mid-epoch layerwise calibration requires config.dali=False")
        if (fractional_group_starts
                and layerwise_poly_strict_recalibrate_before_blend
                and int(getattr(cfg, "gradient_acc", 1)) != 1):
            raise ValueError(
                "Mid-epoch layerwise calibration requires gradient_acc=1")
    layerwise_poly_training_group_limit = int(getattr(
        cfg, "layerwise_poly_training_group_limit",
        len(herpn_conversion_groups)))
    if (herpn_group_schedule
            and not 1 <= layerwise_poly_training_group_limit <= len(
                herpn_conversion_groups)):
        raise ValueError(
            "layerwise_poly_training_group_limit must select between one "
            "and all configured conversion groups")
    layerwise_poly_training_groups = herpn_conversion_groups[
        :layerwise_poly_training_group_limit]
    layerwise_poly_training_group_epochs = herpn_group_epochs[
        :layerwise_poly_training_group_limit]
    if herpn_group_schedule and cfg.resume:
        completed_herpn_groups = (
            int(resumed_completed_herpn_groups)
            if resumed_completed_herpn_groups is not None
            else completed_herpn_groups_from_model(
                backbone.module, herpn_conversion_groups)
        )
    else:
        completed_herpn_groups = sum(
            float(start_epoch) >= float(start) + herpn_transition_epochs
            for start in herpn_group_epochs
        ) if herpn_group_schedule else 0
    completed_herpn_stages = int(math.floor(float(
        backbone.module.herpn_progress.item()) + 1e-6)
    ) if herpn_enabled and not herpn_group_schedule else 0
    completed_precise_relu_stages = int(math.floor(
        precise_relu_progress_at_epoch(
            start_epoch,
            precise_relu_stage_epochs,
            precise_relu_transition_epochs,
        ) + 1e-6
    )) if precise_relu_schedule else 0
    max_steps_per_epoch = int(getattr(cfg, "max_steps_per_epoch", 0))
    scheduled_steps_per_epoch = (
        max_steps_per_epoch if max_steps_per_epoch > 0 else steps_per_epoch)
    affine_group_epochs = tuple(getattr(
        cfg, "affine_group_epochs", ()))
    affine_group_transition_epochs = float(getattr(
        cfg, "affine_group_transition_epochs", 1.0))
    affine_group_calibration_batches = int(getattr(
        cfg, "affine_group_calibration_batches", 0))
    affine_group_schedule = bool(
        affine_group_epochs
        and hasattr(backbone.module, "set_affine_group_blends"))
    if affine_group_schedule:
        affine_group_names = backbone.module.affine_group_names()
        if len(affine_group_epochs) != len(affine_group_names):
            raise ValueError(
                "affine_group_epochs must contain one start for each affine "
                f"group ({len(affine_group_names)}), got "
                f"{len(affine_group_epochs)}")
        if affine_group_calibration_batches <= 0:
            raise ValueError(
                "Grouped affine conversion requires positive "
                "affine_group_calibration_batches")
        if resumed_affine_group_epochs is not None:
            checkpoint_epochs = tuple(resumed_affine_group_epochs)
            if checkpoint_epochs != affine_group_epochs:
                raise ValueError(
                    "Resume checkpoint affine group epochs do not match the "
                    f"config: checkpoint={checkpoint_epochs}, "
                    f"config={affine_group_epochs}")
        if resumed_affine_group_names is not None:
            checkpoint_names = tuple(
                tuple(group) for group in resumed_affine_group_names)
            if checkpoint_names != affine_group_names:
                raise ValueError(
                    "Resume checkpoint affine groups do not match the model")
        if prepbn_decay_steps > 0:
            raise ValueError(
                "Grouped affine conversion cannot also use the global "
                "prepbn decay schedule; set prepbn_decay_steps=0")
        final_affine_blends = simple_gate_blends_at_epoch(
            cfg.num_epoch, affine_group_epochs,
            affine_group_transition_epochs)
        if (getattr(cfg, "affine_group_require_full_conversion", True)
                and min(final_affine_blends, default=0.0) < 1.0):
            raise ValueError(
                "Affine group schedule does not finish before training ends")
        backbone.module.set_affine_group_blends(
            simple_gate_blends_at_epoch(
                start_epoch, affine_group_epochs,
                affine_group_transition_epochs))
    frozen_std_enabled = hasattr(
        backbone.module, "freeze_frozen_std_group")
    frozen_std_group_names = (
        backbone.module.frozen_std_group_names()
        if frozen_std_enabled else ())
    frozen_std_steps = (
        frozen_std_group_steps(
            getattr(cfg, "frozen_std_start_epoch", 1.0),
            getattr(cfg, "frozen_std_group_gap_steps", 200),
            scheduled_steps_per_epoch,
            len(frozen_std_group_names),
        )
        if frozen_std_enabled else ())
    frozen_std_transition_steps = int(getattr(
        cfg, "frozen_std_transition_steps", 0))
    frozen_std_progressive = (
        frozen_std_enabled and frozen_std_transition_steps > 0)
    frozen_std_spatial_margin = float(getattr(
        cfg, "frozen_std_spatial_margin", 1.25))
    frozen_std_max_tail_to_mean_ratio = float(getattr(
        cfg, "frozen_std_max_tail_to_mean_ratio", 8.0))
    frozen_std_max_value = float(getattr(
        cfg, "frozen_std_max_value", 1e4))
    frozen_std_aux_loss_weight = float(getattr(
        cfg, "frozen_std_aux_loss_weight", 0.0))
    if frozen_std_aux_loss_weight < 0.0:
        raise ValueError("frozen_std_aux_loss_weight must be non-negative")
    if frozen_std_aux_loss_weight > 0.0 and not frozen_std_enabled:
        raise ValueError(
            "frozen_std_aux_loss_weight requires a frozen-std backbone")
    if frozen_std_enabled:
        if int(getattr(cfg, "gradient_acc", 1)) != 1:
            raise ValueError(
                "Frozen-std switch steps are optimizer steps and require "
                "gradient_acc=1")
        if not frozen_std_group_names:
            raise RuntimeError("Frozen-std backbone has no conversion groups")
        if frozen_std_steps[0] <= 1:
            raise ValueError(
                "Frozen-std conversion must leave at least one training batch "
                "to initialize every running standard deviation")
        if frozen_std_progressive:
            if not (hasattr(backbone.module, "begin_frozen_std_group")
                    and hasattr(backbone.module,
                                "set_frozen_std_group_blend")):
                raise TypeError(
                    "frozen_std_transition_steps requires a progressive "
                    "frozen-std backbone")
            group_gap = int(getattr(
                cfg, "frozen_std_group_gap_steps", 200))
            if group_gap < frozen_std_transition_steps:
                raise ValueError(
                    "frozen_std_group_gap_steps must be at least "
                    "frozen_std_transition_steps so only one LayerNorm site "
                    "transitions at a time")
            if frozen_std_spatial_margin < 1.0:
                raise ValueError(
                    "frozen_std_spatial_margin must be at least 1")
            if frozen_std_max_tail_to_mean_ratio < 0.0:
                raise ValueError(
                    "frozen_std_max_tail_to_mean_ratio must be non-negative; "
                    "zero disables the optional fidelity guard")
            if frozen_std_max_value <= 0.0:
                raise ValueError("frozen_std_max_value must be positive")
        total_scheduled_steps = scheduled_steps_per_epoch * int(cfg.num_epoch)
        last_conversion_step = frozen_std_steps[-1] + (
            frozen_std_transition_steps if frozen_std_progressive else 0)
        if (getattr(cfg, "frozen_std_require_full_conversion", True)
                and last_conversion_step > total_scheduled_steps):
            raise ValueError(
                "Frozen-std schedule does not finish before training ends: "
                f"last_conversion={last_conversion_step}, "
                f"total_steps={total_scheduled_steps}")
        if resumed_frozen_std_group_names is not None:
            checkpoint_names = tuple(
                tuple(group) for group in resumed_frozen_std_group_names)
            if checkpoint_names != frozen_std_group_names:
                raise ValueError(
                    "Resume checkpoint frozen-std groups do not match the model")
        if resumed_frozen_std_group_steps is not None:
            checkpoint_steps = tuple(
                int(step) for step in resumed_frozen_std_group_steps)
            if checkpoint_steps != frozen_std_steps:
                raise ValueError(
                    "Resume checkpoint frozen-std steps do not match the "
                    f"config: checkpoint={checkpoint_steps}, "
                    f"config={frozen_std_steps}")
        if (resumed_frozen_std_transition_steps is not None
                and int(resumed_frozen_std_transition_steps)
                != frozen_std_transition_steps):
            raise ValueError(
                "Resume checkpoint frozen-std transition length does not "
                "match the config")
        completed_frozen_std_groups = (
            backbone.module.frozen_std_frozen_count())
        if (resumed_completed_frozen_std_groups is not None
                and int(resumed_completed_frozen_std_groups)
                != completed_frozen_std_groups):
            raise ValueError(
                "Resume checkpoint frozen-std completion metadata does not "
                "match its model buffers")
        expected_completed = sum(
            global_step >= step + (
                frozen_std_transition_steps if frozen_std_progressive else 0)
            for step in frozen_std_steps)
        if completed_frozen_std_groups != expected_completed:
            raise ValueError(
                "Frozen-std checkpoint state is inconsistent with global_step: "
                f"model={completed_frozen_std_groups}, "
                f"schedule={expected_completed}, global_step={global_step}")
        if frozen_std_progressive:
            actual_blends = backbone.module.frozen_std_group_blends()
            expected_blends = tuple(
                0.0 if global_step < start else
                min(1.0, (global_step - start)
                    / frozen_std_transition_steps)
                for start in frozen_std_steps)
            if any(abs(actual - expected) > 1e-6
                   for actual, expected in zip(
                       actual_blends, expected_blends)):
                raise ValueError(
                    "Frozen-std checkpoint blends are inconsistent with "
                    f"global_step={global_step}: model={actual_blends}, "
                    f"schedule={expected_blends}")
            for group_index, start in enumerate(frozen_std_steps):
                should_have_started = global_step >= start
                if (backbone.module.frozen_std_group_started(group_index)
                        != should_have_started):
                    raise ValueError(
                        "Frozen-std checkpoint transition state is "
                        f"inconsistent for group {group_index + 1}")
        backbone.module.set_frozen_std_auxiliary_loss(
            frozen_std_aux_loss_weight > 0.0)
        if rank == 0:
            if frozen_std_progressive:
                logging.info(
                    "Progressive spatial frozen-std schedule: %d LayerNorm "
                    "sites, first start=%d, last completion=%d, gap=%d, "
                    "transition=%d, margin=%.4g, tail/mean limit=%.4g, "
                    "magnitude limit=%.4g, already converted=%d",
                    len(frozen_std_group_names), frozen_std_steps[0],
                    last_conversion_step,
                    int(getattr(cfg, "frozen_std_group_gap_steps", 200)),
                    frozen_std_transition_steps, frozen_std_spatial_margin,
                    frozen_std_max_tail_to_mean_ratio, frozen_std_max_value,
                    completed_frozen_std_groups)
            else:
                logging.info(
                    "Frozen-std hard-switch schedule: %d LayerNorm sites, "
                    "first step=%d, last step=%d, gap=%d, already "
                    "converted=%d, auxiliary weight=%.6g",
                    len(frozen_std_group_names), frozen_std_steps[0],
                    frozen_std_steps[-1],
                    int(getattr(cfg, "frozen_std_group_gap_steps", 200)),
                    completed_frozen_std_groups, frozen_std_aux_loss_weight)
    else:
        completed_frozen_std_groups = 0
    simple_gate_stats_interval = int(getattr(
        cfg, "simple_gate_stats_interval", 0))
    simple_gate_enabled = hasattr(
        backbone.module, "set_simple_gate_instrumentation")
    simple_gate_progressive = hasattr(
        backbone.module, "set_simple_gate_blends")
    simple_gate_group_epochs = tuple(getattr(
        cfg, "simple_gate_group_epochs", ()))
    simple_gate_transition_epochs = float(getattr(
        cfg, "simple_gate_transition_epochs", 1.0))
    simple_gate_distill_loss_weight = float(getattr(
        cfg, "simple_gate_distill_loss_weight", 0.0))
    simple_gate_range_loss_weight = float(getattr(
        cfg, "simple_gate_range_loss_weight", 0.0))
    simple_gate_lr_multiplier = float(getattr(
        cfg, "simple_gate_lr_multiplier", 1.0))
    if simple_gate_lr_multiplier <= 0.0:
        raise ValueError("simple_gate_lr_multiplier must be positive")
    simple_gate_schedule = bool(
        simple_gate_progressive and simple_gate_group_epochs)
    if simple_gate_schedule:
        if (simple_gate_distill_loss_weight <= 0
                and simple_gate_range_loss_weight <= 0):
            raise ValueError(
                "Progressive SimpleGate conversion needs an auxiliary loss "
                "so the multiplier half remains in the DDP graph before its "
                "blend becomes nonzero")
        gate_groups = backbone.module.simple_gate_group_names()
        if len(simple_gate_group_epochs) != len(gate_groups):
            raise ValueError(
                "simple_gate_group_epochs must contain one start for each "
                f"SimpleGate group ({len(gate_groups)}), got "
                f"{len(simple_gate_group_epochs)}")
        first_gate_step = int(simple_gate_group_epochs[0] * steps_per_epoch)
        if first_gate_step < prepbn_decay_steps:
            raise ValueError(
                "SimpleGate conversion overlaps RepBatchNorm transition: "
                f"first_gate_step={first_gate_step}, "
                f"prepbn_decay_steps={prepbn_decay_steps}")
        final_gate_blends = simple_gate_blends_at_epoch(
            cfg.num_epoch, simple_gate_group_epochs,
            simple_gate_transition_epochs)
        if (getattr(cfg, "simple_gate_require_full_conversion", True)
                and min(final_gate_blends, default=0.0) < 1.0):
            raise ValueError(
                "SimpleGate schedule does not finish before training ends")
        set_simple_gate_blends(
            backbone.module,
            simple_gate_blends_at_epoch(
                start_epoch, simple_gate_group_epochs,
                simple_gate_transition_epochs),
        )
    if simple_gate_progressive and simple_gate_current_group_auxiliary:
        initial_blends = simple_gate_blends_at_epoch(
            start_epoch, simple_gate_group_epochs,
            simple_gate_transition_epochs)
        initial_group = next(
            (index for index, blend in enumerate(initial_blends)
             if blend < 1.0),
            None,
        )
        backbone.module.set_simple_gate_auxiliary_groups(
            () if initial_group is None else (initial_group,))
    elif simple_gate_progressive:
        backbone.module.set_simple_gate_auxiliary_losses(
            simple_gate_distill_loss_weight > 0
            or simple_gate_range_loss_weight > 0)
    completed_simple_gate_groups = sum(
        float(start_epoch) > float(start) + simple_gate_transition_epochs
        for start in simple_gate_group_epochs
    ) if simple_gate_schedule else 0
    if resumed_completed_simple_gate_groups is not None:
        completed_simple_gate_groups = int(
            resumed_completed_simple_gate_groups)
        if not 0 <= completed_simple_gate_groups <= len(
                simple_gate_group_epochs):
            raise ValueError(
                "Invalid completed_simple_gate_groups in checkpoint: "
                f"{completed_simple_gate_groups}")
    # A completed SimpleGate group proves that the one-time post-RepBN
    # recalibration already ran before that group started.  Do not reset all
    # BN statistics again when resuming a later, partially blended group.
    repbn_gate_recalibrated = bool(
        resumed_repbn_gate_recalibrated
        if resumed_repbn_gate_recalibrated is not None
        else cfg.resume and completed_simple_gate_groups > 0)
    if repbn_gate_recalibrated and rank == 0:
        logging.info(
            "Resume checkpoint records post-RepBatchNorm recalibration "
            "complete with %d completed SimpleGate group(s); skipping it",
            completed_simple_gate_groups)
    simple_gate_repbn_recalibration_batches = int(getattr(
        cfg, "simple_gate_repbn_recalibration_batches", 0))
    simple_gate_verify_after_repbn = bool(getattr(
        cfg, "simple_gate_verify_after_repbn", False))
    simple_gate_group_bn_recalibration_batches = int(getattr(
        cfg, "simple_gate_group_bn_recalibration_batches", 0))
    simple_gate_verify_after_group = bool(getattr(
        cfg, "simple_gate_verify_after_group", False))
    simple_gate_save_after_group = bool(getattr(
        cfg, "simple_gate_save_after_group", False))
    if ((simple_gate_verify_after_group or simple_gate_save_after_group)
            and simple_gate_group_bn_recalibration_batches <= 0):
        raise ValueError(
            "SimpleGate group verification/checkpointing requires "
            "simple_gate_group_bn_recalibration_batches > 0")
    last_simple_gate_snapshot = {}
    last_simple_gate_snapshot_step = None
    last_nf_snapshot = {}
    last_nf_snapshot_step = None
    max_nonfinite_embedding_skips = int(getattr(
        cfg, "max_nonfinite_embedding_skips", 0))
    nonfinite_embedding_skips = 0
    max_nonfinite_loss_skips = int(getattr(
        cfg, "max_nonfinite_loss_skips", 0))
    nonfinite_loss_skips = 0
    if max_nonfinite_embedding_skips < 0:
        raise ValueError("max_nonfinite_embedding_skips must be non-negative")
    if max_nonfinite_loss_skips < 0:
        raise ValueError("max_nonfinite_loss_skips must be non-negative")
    validate_after_prepbn_transition = bool(getattr(
        cfg, "validate_after_prepbn_transition", False))
    layerwise_tail_replay_indices = ()
    layerwise_tail_replay_loader = None
    layerwise_tail_replay_iterator = None
    layerwise_tail_replay_epoch = 0
    fixed_tail_replay_loader = None
    fixed_tail_replay_iterator = None
    fixed_tail_replay_epoch = 0

    fixed_tail_replay_file = str(getattr(
        cfg, "fixed_tail_replay_file", ""))
    fixed_tail_replay_batch_size = int(getattr(
        cfg, "fixed_tail_replay_batch_size", 0))
    fixed_tail_replay_workers = int(getattr(
        cfg, "fixed_tail_replay_workers", 0))
    fixed_tail_replay_priority_count = int(getattr(
        cfg, "fixed_tail_replay_priority_count", 0))
    fixed_tail_replay_priority_repeats = int(getattr(
        cfg, "fixed_tail_replay_priority_repeats", 1))
    fixed_tail_replay_orientations_key = str(getattr(
        cfg, "fixed_tail_replay_orientations_key", ""))
    if fixed_tail_replay_file:
        if fixed_tail_replay_batch_size <= 0:
            raise ValueError(
                "fixed_tail_replay_file requires a positive replay batch size")
        if fixed_tail_replay_orientations_key:
            replay_orientations = load_fixed_tail_replay_orientations(
                fixed_tail_replay_file,
                fixed_tail_replay_orientations_key,
            )
            oriented_dataset = DatasetWithIndex(
                train_loader.dataset, both_orientations=True)
            replay_indices = tuple(
                2 * source_index + orientation
                for source_index, orientation in replay_orientations
            )
            fixed_subset = Subset(oriented_dataset, replay_indices)
            fixed_indices = tuple(dict.fromkeys(
                source_index for source_index, _ in replay_orientations))
        else:
            fixed_indices = load_fixed_tail_replay_indices(
                fixed_tail_replay_file)
            replay_indices = prioritized_tail_replay_indices(
                fixed_indices,
                fixed_tail_replay_priority_count,
                fixed_tail_replay_priority_repeats,
            )
            fixed_subset = Subset(train_loader.dataset, replay_indices)
        fixed_sampler = TorchDistributedSampler(
            fixed_subset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=int(cfg.seed),
            drop_last=False,
        )
        fixed_tail_replay_loader = DataLoader(
            fixed_subset,
            batch_size=fixed_tail_replay_batch_size,
            sampler=fixed_sampler,
            num_workers=fixed_tail_replay_workers,
            pin_memory=True,
            drop_last=False,
            worker_init_fn=train_loader.worker_init_fn,
        )
        fixed_sampler.set_epoch(0)
        fixed_tail_replay_iterator = iter(fixed_tail_replay_loader)
        if rank == 0:
            logging.info(
                "Configured fixed hard-tail replay: unique_indices=%d "
                "replay_entries=%d priority_count=%d priority_repeats=%d "
                "orientations_key=%s "
                "batch_size_per_rank=%d manifest=%s",
                len(fixed_indices), len(replay_indices),
                min(fixed_tail_replay_priority_count, len(fixed_indices)),
                fixed_tail_replay_priority_repeats,
                fixed_tail_replay_orientations_key or "random",
                fixed_tail_replay_batch_size, fixed_tail_replay_file)

    def next_fixed_tail_replay_batch():
        nonlocal fixed_tail_replay_iterator
        nonlocal fixed_tail_replay_epoch
        if fixed_tail_replay_loader is None:
            return None
        try:
            return next(fixed_tail_replay_iterator)
        except StopIteration:
            fixed_tail_replay_epoch += 1
            fixed_tail_replay_loader.sampler.set_epoch(
                fixed_tail_replay_epoch)
            fixed_tail_replay_iterator = iter(fixed_tail_replay_loader)
            return next(fixed_tail_replay_iterator)

    def configure_layerwise_tail_replay(results):
        """Build a distributed replay stream from calibration extrema."""
        nonlocal layerwise_tail_replay_indices
        nonlocal layerwise_tail_replay_loader
        nonlocal layerwise_tail_replay_iterator
        nonlocal layerwise_tail_replay_epoch
        if layerwise_poly_tail_replay_batch_size <= 0:
            return
        result_tail_groups = tuple(
            tuple(result.get("tail_indices", ())) for result in results)
        new_indices = tuple(dict.fromkeys(
            index for group in result_tail_groups for index in group))
        if not new_indices:
            raise RuntimeError(
                "Tail replay was enabled but calibration returned no indices")
        # A later singleton's provisional scan must not discard extrema from
        # the already accepted polynomial prefix.  Keep a stable union across
        # restored manifests and newly calibrated activations.
        indices = merge_tail_replay_indices(
            layerwise_tail_replay_indices,
            result_tail_groups,
            layerwise_poly_tail_replay_extra_indices,
        )
        replay_indices = prioritized_tail_replay_indices(
            indices,
            layerwise_poly_tail_replay_priority_count,
            layerwise_poly_tail_replay_priority_repeats,
        )
        subset = Subset(train_loader.dataset, replay_indices)
        sampler = TorchDistributedSampler(
            subset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=int(cfg.seed),
            drop_last=False,
        )
        layerwise_tail_replay_indices = indices
        layerwise_tail_replay_loader = DataLoader(
            subset,
            batch_size=layerwise_poly_tail_replay_batch_size,
            sampler=sampler,
            num_workers=layerwise_poly_tail_replay_workers,
            pin_memory=True,
            drop_last=False,
            worker_init_fn=train_loader.worker_init_fn,
        )
        layerwise_tail_replay_epoch = 0
        sampler.set_epoch(layerwise_tail_replay_epoch)
        layerwise_tail_replay_iterator = iter(layerwise_tail_replay_loader)
        if rank == 0:
            logging.info(
                "Configured layerwise tail replay: unique_indices=%d "
                "weighted_entries=%d priority_count=%d priority_repeats=%d "
                "batch_size_per_rank=%d",
                len(indices), len(replay_indices),
                min(layerwise_poly_tail_replay_priority_count, len(indices)),
                layerwise_poly_tail_replay_priority_repeats,
                layerwise_poly_tail_replay_batch_size)

    def next_layerwise_tail_replay_batch():
        nonlocal layerwise_tail_replay_iterator
        nonlocal layerwise_tail_replay_epoch
        if layerwise_tail_replay_loader is None:
            return None
        try:
            return next(layerwise_tail_replay_iterator)
        except StopIteration:
            layerwise_tail_replay_epoch += 1
            layerwise_tail_replay_loader.sampler.set_epoch(
                layerwise_tail_replay_epoch)
            layerwise_tail_replay_iterator = iter(
                layerwise_tail_replay_loader)
            return next(layerwise_tail_replay_iterator)

    def calibrate_layerwise_poly_group(
            target_names, *, allow_recalibration=False, provisional=False):
        target_names = tuple(target_names)
        if rank == 0:
            logging.info(
                "Starting layerwise polynomial %s pass: %s",
                ("provisional calibration" if provisional else
                 "strict recalibration" if allow_recalibration else
                 "calibration"),
                ", ".join(target_names))
        try:
            results = calibrate_layerwise_poly_input_scales(
                backbone,
                layerwise_poly_range_loader,
                target_names,
                layerwise_poly_range_batches,
                layerwise_poly_range_margin,
                layerwise_poly_min_scale,
                global_step,
                dali=cfg.dali,
                quantile=layerwise_poly_range_quantile,
                quantile_samples=layerwise_poly_quantile_samples,
                holdout_fraction=layerwise_poly_range_holdout_fraction,
                max_tail_ratio=layerwise_poly_max_tail_ratio,
                max_scale_growth=layerwise_poly_max_scale_growth,
                max_input_scale=layerwise_poly_max_input_scale,
                allow_recalibration=allow_recalibration,
                allow_provisional_tail=provisional,
                enforce_tail_scale_floor=(
                    allow_recalibration
                    and not provisional
                    and layerwise_poly_strict_tail_scale_floor),
                tail_scale_floor_margin=layerwise_poly_tail_scale_floor_margin,
                max_tail_scale_expansion=(
                    layerwise_poly_max_tail_scale_expansion),
                progress_interval_batches=(
                    layerwise_poly_calibration_log_interval),
                require_full_containment=(
                    layerwise_poly_require_full_containment),
                preserve_existing_scale=(
                    allow_recalibration
                    and layerwise_poly_freeze_containment_interval),
                tail_topk=layerwise_poly_tail_topk,
            )
        except FloatingPointError as error:
            if rank == 0:
                logging.error(
                    "Layerwise polynomial calibration failed for %s: %s",
                    ", ".join(target_names), error)
            raise
        if rank == 0:
            logging.info(
                "Layerwise polynomial group %s in one pass: %s",
                "provisionally calibrated" if provisional else "calibrated",
                ", ".join(target_names))
            for result in results:
                log = (
                    logging.warning if result["provisional"] else logging.info)
                log(
                    "Layerwise polynomial interval %s: %s robust "
                    "q=%.6g absmax=%.7g observed_absmax=%.7g "
                    "tail_ratio=%.5g margin=%.4g scale=%.7g "
                    "scale_growth=%s tail_floor=%s expansion=%.5g "
                    "floor_applied=%s batches/rank=%d holdout/rank=%d "
                    "max_source=%s violations=%s",
                    "PROVISIONAL" if result["provisional"] else "calibrated",
                    result["activation"],
                    result["quantile"],
                    result["robust_absmax"],
                    result["observed_absmax"],
                    result["tail_ratio"],
                    result["margin"],
                    result["input_scale"],
                    (f'{result["scale_growth"]:.5g}'
                     if result["scale_growth"] is not None else "n/a"),
                    (f'{result["tail_scale_floor"]:.7g}'
                     if result["tail_scale_floor"] is not None else "n/a"),
                    result["tail_scale_expansion"],
                    result["tail_floor_applied"],
                    result["batches_per_rank"],
                    result["holdout_batches_per_rank"],
                    result["source"],
                    ", ".join(result["violations"]) or "none",
                )
                if result["tail_indices"]:
                    logging.info(
                        "Layerwise polynomial tail replay indices for %s: %s",
                        result["activation"],
                        ",".join(str(index) for index in result["tail_indices"]),
                    )
                    tail_path = os.path.join(
                        cfg.output,
                        "tail_replay_"
                        + result["activation"].replace(".", "_")
                        + ".json",
                    )
                    temporary_tail_path = tail_path + ".tmp"
                    with open(temporary_tail_path, "w") as tail_handle:
                        json.dump({
                            "activation": result["activation"],
                            "global_step": global_step,
                            "input_scale": result["input_scale"],
                            "observed_absmax": result["observed_absmax"],
                            "tail_ratio": result["tail_ratio"],
                            "source": result["source"],
                            "dataset_indices": list(result["tail_indices"]),
                        }, tail_handle, indent=2, sort_keys=True)
                        tail_handle.write("\n")
                    os.replace(temporary_tail_path, tail_path)
                if summary_writer is not None:
                    summary_writer.add_scalar(
                        "LayerwisePoly/InputScale/"
                        + result["activation"].replace(".", "/"),
                        result["input_scale"],
                        global_step,
                    )
                    summary_writer.add_scalar(
                        "LayerwisePoly/TailRatio/"
                        + result["activation"].replace(".", "/"),
                        result["tail_ratio"],
                        global_step,
                    )
                    summary_writer.add_scalar(
                        "LayerwisePoly/Provisional/"
                        + result["activation"].replace(".", "/"),
                        float(result["provisional"]),
                        global_step,
                    )
                    summary_writer.add_scalar(
                        "LayerwisePoly/TailScaleExpansion/"
                        + result["activation"].replace(".", "/"),
                        result["tail_scale_expansion"],
                        global_step,
                    )
        configure_layerwise_tail_replay(results)
        return results

    def strictly_calibrate_layerwise_poly_group(target_names, group_index):
        target_names = tuple(target_names)
        if layerwise_poly_require_full_containment:
            # Re-audit the complete polynomial prefix against its immutable
            # intervals. Upstream task/blend updates are not allowed to make a
            # previously accepted activation unsafe while conditioning the
            # next singleton.
            model_order = tuple(
                backbone.module.layerwise_poly_activation_names())
            last_index = model_order.index(target_names[-1])
            activations = dict(
                backbone.module.named_progressive_activations())
            prefix_names = tuple(
                name for name in model_order[:last_index + 1]
                if activations[name]._scale_is_calibrated)
            results = calibrate_layerwise_poly_group(
                prefix_names, allow_recalibration=True, provisional=False)
            if layerwise_poly_verify_singleton_boundary:
                def verify_contained_singleton(names):
                    return verify_layerwise_poly_group_boundary(
                        backbone,
                        layerwise_poly_range_loader,
                        names,
                        layerwise_poly_range_batches,
                        global_step,
                        layerwise_poly_max_input_scale,
                        dali=cfg.dali,
                        progress_interval_batches=(
                            layerwise_poly_calibration_log_interval),
                    )

                _, verification = causally_calibrate_polynomial_group(
                    backbone.module,
                    target_names,
                    lambda name, member_index, member_count: [],
                    verify_contained_singleton,
                )
                if rank == 0:
                    logging.info(
                        "Full-containment boundary passed for group %d/%d: "
                        "boundary=%s absmax=%.7g batches/rank=%d",
                        group_index + 1, len(herpn_conversion_groups),
                        verification["boundary"], verification["absmax"],
                        verification["batches_per_rank"])
            return results
        use_causal_boundary = (
            layerwise_poly_causal_strict_calibration
            and (len(target_names) > 1
                 or layerwise_poly_verify_singleton_boundary)
        )
        if not use_causal_boundary:
            return calibrate_layerwise_poly_group(
                target_names, allow_recalibration=True, provisional=False)

        if rank == 0:
            logging.info(
                "Starting causal strict calibration for group %d/%d: %s",
                group_index + 1, len(herpn_conversion_groups),
                ", ".join(target_names))

        def calibrate_one(name, member_index, member_count):
            if rank == 0:
                logging.info(
                    "Causal strict calibration member %d/%d: %s",
                    member_index + 1, member_count, name)
            return calibrate_layerwise_poly_group(
                (name,), allow_recalibration=True, provisional=False)

        def verify_group(names):
            if rank == 0:
                logging.info(
                    "Verifying fully polynomial group %d/%d boundary: %s",
                    group_index + 1, len(herpn_conversion_groups),
                    ", ".join(names))
            return verify_layerwise_poly_group_boundary(
                backbone,
                layerwise_poly_range_loader,
                names,
                layerwise_poly_range_batches,
                global_step,
                layerwise_poly_max_input_scale,
                dali=cfg.dali,
                progress_interval_batches=(
                    layerwise_poly_calibration_log_interval),
            )

        results, verification = causally_calibrate_polynomial_group(
            backbone.module, target_names, calibrate_one, verify_group)
        if rank == 0:
            logging.info(
                "Causal strict calibration passed for group %d/%d: "
                "boundary=%s absmax=%.7g batches/rank=%d",
                group_index + 1, len(herpn_conversion_groups),
                verification["boundary"], verification["absmax"],
                verification["batches_per_rank"])
        return results

    strictly_revalidated_layerwise_groups = set()

    def strictly_revalidate_layerwise_group_before_blend(group_index):
        if group_index in strictly_revalidated_layerwise_groups:
            return []
        target_names = herpn_conversion_groups[group_index]
        if rank == 0:
            logging.info(
                "Strictly revalidating conditioned layerwise polynomial "
                "group %d/%d before blend: %s",
                group_index + 1,
                len(herpn_conversion_groups),
                ", ".join(target_names),
            )
        try:
            results = strictly_calibrate_layerwise_poly_group(
                target_names, group_index)
        except FloatingPointError as error:
            raise FloatingPointError(
                "Layerwise polynomial range conditioning did not make "
                f"group {group_index + 1} safe before its blend. Keep the "
                "saved completed-group checkpoint and increase the local-fit "
                "fraction before this group's blend."
            ) from error
        strictly_revalidated_layerwise_groups.add(group_index)
        return results

    def calibrate_next_layerwise_poly_group(*, provisional=False):
        if not layerwise_poly_enabled:
            return []
        pending = set(backbone.module.uncalibrated_layerwise_poly_names())
        if not pending:
            return []
        target_names = None
        for group in layerwise_poly_training_groups:
            uncalibrated = tuple(name for name in group if name in pending)
            if uncalibrated:
                target_names = uncalibrated
                break
        if not target_names:
            # A selective run may intentionally leave activations beyond its
            # requested conversion frontier as PReLUs.
            return []
        return calibrate_layerwise_poly_group(
            target_names, provisional=provisional)

    # Profile the first pending group before local-fit warmup.  A normal
    # resume already carries that pending interval in the backbone state.  A
    # deliberately expanded hard-containment frontier (for example, resuming
    # a one-group proof with a two-group config) does not: calibrate its first
    # newly exposed singleton now so the remaining local-fit epochs can replay
    # the measured tails.  Never discover an interval at or after its blend
    # boundary, because that would silently skip the conditioning phase.
    should_calibrate_initial_group = layerwise_poly_enabled and (
        not cfg.resume
        or (
            layerwise_poly_require_full_containment
            and completed_herpn_groups
            < layerwise_poly_training_group_limit
            and pending_group_requires_calibration(
                backbone.module.uncalibrated_layerwise_poly_names(),
                layerwise_poly_training_groups,
                completed_herpn_groups,
            )
        )
    )
    if (cfg.resume and layerwise_poly_enabled
            and layerwise_poly_tail_replay_batch_size > 0):
        model_order = tuple(
            backbone.module.layerwise_poly_activation_names())
        activations = dict(
            backbone.module.named_progressive_activations())
        calibrated_names = tuple(
            name for name in model_order
            if activations[name]._scale_is_calibrated)
        calibrated_prefix = calibrated_conversion_prefix(
            model_order,
            calibrated_names,
            layerwise_poly_training_groups,
        )
        if calibrated_prefix:
            manifest_results = load_tail_replay_manifests(
                cfg.output, calibrated_prefix)
            for result in manifest_results:
                checkpoint_scale = float(
                    activations[result["activation"]].input_scale.item())
                if not math.isclose(
                        result["input_scale"], checkpoint_scale,
                        rel_tol=1e-7, abs_tol=0.0):
                    raise ValueError(
                        "Tail replay manifest interval does not match the "
                        "resume checkpoint: activation="
                        f"{result['activation']}, manifest="
                        f"{result['input_scale']:.9g}, checkpoint="
                        f"{checkpoint_scale:.9g}")
            configure_layerwise_tail_replay(manifest_results)
            if rank == 0:
                logging.info(
                    "Restored rare-tail replay manifests for calibrated "
                    "prefix: %s",
                    ", ".join(calibrated_prefix))
    if should_calibrate_initial_group:
        if cfg.resume:
            pending_group_index = completed_herpn_groups
            pending_start = float(
                layerwise_poly_training_group_epochs[pending_group_index])
            if float(start_epoch) >= pending_start:
                raise RuntimeError(
                    "Cannot expand a hard-containment conversion frontier at "
                    "or after the pending blend boundary: "
                    f"start_epoch={start_epoch}, group="
                    f"{pending_group_index + 1}, blend_start={pending_start:g}. "
                    "Move that group's start later so provisional calibration "
                    "and rare-tail conditioning occur before strict blend "
                    "revalidation."
                )
            if rank == 0:
                logging.info(
                    "Expanded hard-containment frontier on resume; "
                    "provisionally calibrating pending group %d/%d before "
                    "local-fit conditioning",
                    pending_group_index + 1,
                    layerwise_poly_training_group_limit,
                )
        if isinstance(layerwise_poly_range_loader, DataLoader):
            layerwise_poly_range_loader.sampler.set_epoch(start_epoch)
        calibrate_next_layerwise_poly_group(
            provisional=layerwise_poly_initial_calibration_provisional)

    for epoch in range(start_epoch, cfg.num_epoch):

        if pillar_enabled:
            pillar_effective_coefficient, pillar_effective_exponent = (
                pillar_regularization_at_epoch(
                    epoch,
                    pillar_target_coefficient,
                    pillar_target_exponent,
                    warmup=pillar_regularization_warmup,
                )
            )
            backbone.module.set_pillar_regularization_exponent(
                pillar_effective_exponent)
            pillar_task_loss_weight = pillar_task_loss_weight_at_epoch(
                epoch, pillar_range_only_epochs)
            if rank == 0:
                logging.info(
                    "PILLAR epoch %d: beta=%g gamma=%d "
                    "task_loss_weight=%g",
                    epoch, pillar_effective_coefficient,
                    pillar_effective_exponent, pillar_task_loss_weight,
                )

        if isinstance(train_loader, DataLoader):
            train_loader.sampler.set_epoch(epoch)
        if (layerwise_poly_enabled
                and isinstance(layerwise_poly_range_loader, DataLoader)):
            layerwise_poly_range_loader.sampler.set_epoch(epoch)
        if affine_group_schedule:
            epoch_affine_blends = simple_gate_blends_at_epoch(
                epoch, affine_group_epochs,
                affine_group_transition_epochs)
            backbone.module.set_affine_group_blends(epoch_affine_blends)
            starting_groups = [
                index for index, start in enumerate(affine_group_epochs)
                if abs(float(epoch) - float(start)) < 1e-9
            ]
            if len(starting_groups) > 1:
                raise RuntimeError(
                    "Only one affine group may start in an epoch because each "
                    "group requires current-distribution calibration")
            if starting_groups:
                group_index = starting_groups[0]
                completed, diagnostics = calibrate_affine_normalization(
                    backbone.module,
                    affine_calibration_loader,
                    affine_group_calibration_batches,
                    ridge=float(getattr(
                        cfg, "affine_calibration_ridge", 1e-6)),
                    dali=cfg.dali,
                    group_index=group_index,
                )
                if rank == 0:
                    worst = max(
                        diagnostics,
                        key=lambda item: item["relative_rmse_max"])
                    logging.info(
                        "Calibrated affine group %d/%d (%s) with %d batches "
                        "per rank; worst=%s relative_rmse_max=%.6g "
                        "scale_absmax=%.6g bias_absmax=%.6g",
                        group_index + 1,
                        len(affine_group_names),
                        ", ".join(affine_group_names[group_index]),
                        completed,
                        worst["name"],
                        worst["relative_rmse_max"],
                        max(item["scale_absmax"] for item in diagnostics),
                        max(item["bias_absmax"] for item in diagnostics),
                    )
        if (simple_gate_schedule and not repbn_gate_recalibrated
                and prepbn_transition_complete(global_step, prepbn_decay_steps)):
            # Freeze the normalization graph at pure RepBN and evaluate it
            # before the first multiplication gate enters the main path.
            set_prepbn_progress(
                backbone.module, prepbn_decay_steps, prepbn_decay_steps)
            set_simple_gate_blends(
                backbone.module,
                simple_gate_blends_at_epoch(
                    epoch, simple_gate_group_epochs,
                    simple_gate_transition_epochs),
            )
            calibrated = recalibrate_batchnorm_batches(
                backbone, train_loader,
                simple_gate_repbn_recalibration_batches,
                global_step, "post-RepBatchNorm")
            if cfg.dali:
                train_loader.reset()
            if rank == 0:
                logging.info(
                    "RepBatchNorm transition complete; refreshed BN with %d "
                    "batches before SimpleGate conversion", calibrated)
            if simple_gate_verify_after_repbn:
                callback_verification(global_step, backbone.module)
            repbn_gate_recalibrated = True
        if simple_gate_schedule:
            epoch_gate_blends = simple_gate_blends_at_epoch(
                epoch, simple_gate_group_epochs,
                simple_gate_transition_epochs)
            set_simple_gate_blends(backbone.module, epoch_gate_blends)
            newly_completed_gates = sum(
                float(epoch) >= float(start) + simple_gate_transition_epochs
                for start in simple_gate_group_epochs)
            if newly_completed_gates > completed_simple_gate_groups:
                if newly_completed_gates != completed_simple_gate_groups + 1:
                    raise RuntimeError(
                        "SimpleGate schedule crossed more than one group "
                        "completion boundary in a single epoch; use epoch-aligned "
                        "group transitions so every group can be recalibrated "
                        "and checkpointed before the next group begins")
                finalized_gate_blends = list(epoch_gate_blends)
                finalized_gate_blends[newly_completed_gates - 1] = 1.0
                if any(
                        blend > 0.0
                        for blend in finalized_gate_blends[
                            newly_completed_gates:]):
                    raise RuntimeError(
                        "A later SimpleGate group started before the completed "
                        "group's recalibration boundary")
                finalized_gate_blends = tuple(finalized_gate_blends)
                set_simple_gate_blends(
                    backbone.module, finalized_gate_blends)
                if rank == 0:
                    logging.info(
                        "SimpleGate group %d/%d completed; forcing blends=%s",
                        newly_completed_gates, len(simple_gate_group_epochs),
                        ",".join(
                            f"{value:.3f}"
                            for value in finalized_gate_blends))
                if simple_gate_group_bn_recalibration_batches > 0:
                    if rank == 0:
                        logging.info(
                            "Recalibrating all BatchNorm statistics after "
                            "SimpleGate group %d/%d with %d batches per rank",
                            newly_completed_gates,
                            len(simple_gate_group_epochs),
                            simple_gate_group_bn_recalibration_batches)
                    calibrated = recalibrate_batchnorm_batches(
                        backbone,
                        train_loader,
                        simple_gate_group_bn_recalibration_batches,
                        global_step,
                        f"post-SimpleGate group {newly_completed_gates}",
                    )
                    if cfg.dali:
                        train_loader.reset()
                    if rank == 0:
                        logging.info(
                            "SimpleGate group %d/%d BatchNorm recalibration "
                            "complete; refreshed %d batches per rank",
                            newly_completed_gates,
                            len(simple_gate_group_epochs),
                            calibrated)
                    if simple_gate_verify_after_group:
                        callback_verification(global_step, backbone.module)
                    if simple_gate_save_after_group and rank == 0:
                        group_checkpoint_path = os.path.join(
                            cfg.output,
                            "model_simple_gate_group_"
                            f"{newly_completed_gates:02d}_bnrecalibrated.pt",
                        )
                        atomic_torch_save(
                            {
                                "state_dict_backbone":
                                    backbone.module.state_dict(),
                                "simple_gate_blends":
                                    finalized_gate_blends,
                                "simple_gate_group":
                                    newly_completed_gates,
                                "simple_gate_group_names":
                                    gate_groups[
                                        newly_completed_gates - 1],
                                "simple_gate_grouping": str(getattr(
                                    cfg, "simple_gate_grouping",
                                    "stage_chunks")),
                                "epoch": epoch,
                                "global_step": global_step,
                                "bn_recalibration_batches_per_rank":
                                    calibrated,
                                "bn_recalibration_world_size":
                                    distributed.get_world_size(),
                            },
                            group_checkpoint_path,
                        )
                        logging.info(
                            "Saved recalibrated SimpleGate group %d "
                            "checkpoint to %s",
                            newly_completed_gates, group_checkpoint_path)
                    if (distributed.is_available()
                            and distributed.is_initialized()):
                        distributed.barrier()
                completed_simple_gate_groups = newly_completed_gates
        if precise_relu_schedule:
            epoch_precise_progress = precise_relu_progress_at_epoch(
                epoch,
                precise_relu_stage_epochs,
                precise_relu_transition_epochs,
            )
            backbone.module.set_polynomial_progress(epoch_precise_progress)
            newly_completed_precise_stages = int(math.floor(
                epoch_precise_progress + 1e-6))
            if (newly_completed_precise_stages
                    > completed_precise_relu_stages):
                if rank == 0:
                    logging.info(
                        "PreciseReLU stage %d/%d completed (%s); "
                        "recalibrating BatchNorm with %d batches",
                        newly_completed_precise_stages,
                        backbone.module.polynomial_transition_count(),
                        backbone.module.polynomial_stage_names()[
                            newly_completed_precise_stages],
                        precise_relu_bn_recalibration_batches,
                    )
                recalibrate_herpn_batchnorm(
                    backbone,
                    herpn_recalibration_loader,
                    precise_relu_bn_recalibration_batches,
                    global_step,
                )
                if cfg.dali:
                    train_loader.reset()
                completed_precise_relu_stages = (
                    newly_completed_precise_stages)
        if herpn_group_schedule:
            epoch_blends = herpn_group_blends_at_epoch(
                epoch, herpn_conversion_groups, herpn_group_epochs,
                herpn_transition_epochs)
            backbone.module.set_herpn_blends(epoch_blends)
            if (layerwise_poly_enabled
                    and layerwise_poly_strict_recalibrate_before_blend):
                starting_group_indices = [
                    group_index
                    for group_index, start in enumerate(herpn_group_epochs)
                    if abs(float(epoch) - float(start)) < 1e-9
                ]
                for group_index in starting_group_indices:
                    strictly_revalidate_layerwise_group_before_blend(
                        group_index)
            newly_completed = sum(
                float(epoch) >= float(start) + herpn_transition_epochs
                for start in herpn_group_epochs)
            if newly_completed > completed_herpn_groups:
                completed_names = [
                    name
                    for group in herpn_conversion_groups[
                        completed_herpn_groups:newly_completed]
                    for name in group
                ]
                if rank == 0:
                    logging.info(
                        "HerPN group %d/%d completed (%s); recalibrating "
                        "BatchNorm with %d batches",
                        newly_completed, len(herpn_conversion_groups),
                        ", ".join(completed_names),
                        herpn_bn_recalibration_batches)
                recalibrate_herpn_batchnorm(
                    backbone, herpn_recalibration_loader,
                    herpn_bn_recalibration_batches,
                    global_step,
                    after_activation_name=(
                        # A multi-activation group changes every BN downstream
                        # of its earliest member, not only BNs after the last.
                        completed_names[0]
                        if layerwise_poly_enabled else None),
                )
                if cfg.dali:
                    train_loader.reset()
                if layerwise_poly_require_full_containment:
                    model_order = tuple(
                        backbone.module.layerwise_poly_activation_names())
                    completed_last_name = completed_names[-1]
                    completed_last_index = model_order.index(
                        completed_last_name)
                    contained_prefix = model_order[:completed_last_index + 1]
                    if rank == 0:
                        logging.info(
                            "Auditing completed polynomial prefix %d/%d "
                            "against immutable approximation intervals",
                            newly_completed, len(herpn_conversion_groups))
                    calibrate_layerwise_poly_group(
                        contained_prefix,
                        allow_recalibration=True,
                        provisional=False,
                    )
                # Persist the known-good completed graph before profiling the
                # next group. A rejected pending interval can never prevent
                # recovery of this BatchNorm-recalibrated boundary.
                if herpn_save_after_group and rank == 0:
                    group_checkpoint_path = os.path.join(
                        cfg.output,
                        "model_herpn_group_"
                        f"{newly_completed:02d}_bnrecalibrated.pt",
                    )
                    serialized_blends = {
                        name: float(activation.blend.item())
                        for name, activation
                        in backbone.module.named_modules()
                        if getattr(
                            activation,
                            "is_progressive_polynomial_activation",
                            False)
                    }
                    atomic_torch_save(
                        {
                            "state_dict_backbone":
                                backbone.module.state_dict(),
                            "herpn_blends": serialized_blends,
                            "herpn_group": newly_completed,
                            "herpn_conversion_groups":
                                herpn_conversion_groups,
                            "epoch": epoch,
                            "global_step": global_step,
                            "bn_recalibration_batches_per_rank":
                                herpn_bn_recalibration_batches,
                            "bn_recalibration_world_size":
                                distributed.get_world_size(),
                        },
                        group_checkpoint_path,
                    )
                    logging.info(
                        "Saved recalibrated HerPN group %d checkpoint to %s",
                        newly_completed, group_checkpoint_path)
                if (herpn_save_after_group
                        and distributed.is_available()
                        and distributed.is_initialized()):
                    distributed.barrier()
                completed_herpn_groups = newly_completed
                if (layerwise_poly_enabled
                        and newly_completed
                        < layerwise_poly_training_group_limit):
                    # Tail-only violations are accepted provisionally while
                    # blend remains zero. The local-fit gap then uses the range
                    # loss to condition upstream convolution/BN weights before
                    # a mandatory strict pass at the blend boundary.
                    calibrate_next_layerwise_poly_group(
                        provisional=layerwise_poly_allow_provisional_tail)
        elif herpn_enabled and herpn_stage_epochs:
            epoch_herpn_progress = herpn_progress_at_epoch(
                epoch, herpn_stage_epochs, herpn_transition_epochs)
            backbone.module.set_herpn_progress(epoch_herpn_progress)
            newly_completed = int(math.floor(epoch_herpn_progress + 1e-6))
            if newly_completed > completed_herpn_stages:
                if rank == 0:
                    logging.info(
                        "HerPN stage %d/5 completed; recalibrating BatchNorm with %d batches",
                        newly_completed, herpn_bn_recalibration_batches)
                recalibrate_herpn_batchnorm(
                    backbone, herpn_recalibration_loader,
                    herpn_bn_recalibration_batches, global_step)
                if cfg.dali:
                    train_loader.reset()
                completed_herpn_stages = newly_completed
        if layerwise_poly_staged_training and rank == 0:
            phase, active_group_index = layerwise_poly_group_phase_at_epoch(
                epoch, layerwise_poly_training_groups,
                layerwise_poly_training_group_epochs,
                herpn_transition_epochs)
            active_names = (
                layerwise_poly_training_groups[active_group_index]
                if active_group_index is not None else ())
            logging.info(
                "Layerwise polynomial phase=%s active_group=%s "
                "backbone_lr_scale=%.4g polynomial_lr_scale=%.4g",
                phase,
                ", ".join(active_names) if active_names else "all",
                (layerwise_poly_blend_backbone_lr_scale
                 if phase == "blend" else
                 layerwise_poly_final_backbone_lr_scale
                 if phase == "final_finetune" else
                 layerwise_poly_local_fit_backbone_lr_scale),
                layerwise_poly_optimizer_lr_scale,
            )
        for step_in_epoch, (img, local_labels) in enumerate(train_loader):
            if max_steps_per_epoch > 0 and step_in_epoch >= max_steps_per_epoch:
                break
            global_step += 1
            if (frozen_std_enabled
                    and completed_frozen_std_groups
                    < len(frozen_std_steps)):
                group_index = completed_frozen_std_groups
                start_step = frozen_std_steps[group_index]
                if frozen_std_progressive:
                    completion_step = (
                        start_step + frozen_std_transition_steps)
                    if global_step == start_step:
                        diagnostics = (
                            backbone.module.begin_frozen_std_group(
                                group_index,
                                distributed=True,
                                margin=frozen_std_spatial_margin,
                                max_tail_to_mean_ratio=(
                                    frozen_std_max_tail_to_mean_ratio),
                                max_frozen_std=frozen_std_max_value,
                            ))
                        if rank == 0:
                            logging.info(
                                "Started spatial frozen-std group %d/%d at "
                                "step %d: %s",
                                group_index + 1,
                                len(frozen_std_group_names), global_step,
                                ", ".join(
                                    f"{item['name']} tail="
                                    f"{item['tail_max']:.6g} frozen_max="
                                    f"{item['frozen_max']:.6g} tail/mean="
                                    f"{item['tail_to_mean_max']:.6g}"
                                    for item in diagnostics),
                            )
                    if start_step <= global_step <= completion_step:
                        if not backbone.module.frozen_std_group_started(
                                group_index):
                            raise RuntimeError(
                                "Missed frozen-std transition start for group "
                                f"{group_index + 1} at step {start_step}")
                        blend = min(
                            1.0,
                            (global_step - start_step)
                            / frozen_std_transition_steps,
                        )
                        backbone.module.set_frozen_std_group_blend(
                            group_index, blend)
                        if global_step == completion_step:
                            completed_frozen_std_groups += 1
                            if rank == 0:
                                logging.info(
                                    "Completed spatial frozen-std group "
                                    "%d/%d at step %d",
                                    completed_frozen_std_groups,
                                    len(frozen_std_group_names), global_step)
                elif global_step == start_step:
                    diagnostics = backbone.module.freeze_frozen_std_group(
                        group_index, distributed=True)
                    completed_frozen_std_groups += 1
                    if rank == 0:
                        logging.info(
                            "Hard-switched frozen-std group %d/%d at step "
                            "%d: %s",
                            completed_frozen_std_groups,
                            len(frozen_std_group_names),
                            global_step,
                            ", ".join(
                                f"{name} std={value:.6g}"
                                for name, value in diagnostics),
                        )
            fractional_epoch = epoch + step_in_epoch / max(
                scheduled_steps_per_epoch, 1)
            if precise_relu_schedule:
                backbone.module.set_polynomial_progress(
                    precise_relu_progress_at_epoch(
                        fractional_epoch,
                        precise_relu_stage_epochs,
                        precise_relu_transition_epochs,
                    ))
            if nf_modulation_schedule:
                nf_modulation_setter(
                    simple_gate_blends_at_epoch(
                        fractional_epoch,
                        nf_modulation_group_epochs,
                        nf_modulation_transition_epochs,
                    ),
                    order=nf_modulation_order,
                )
            if affine_group_schedule:
                backbone.module.set_affine_group_blends(
                    simple_gate_blends_at_epoch(
                        fractional_epoch,
                        affine_group_epochs,
                        affine_group_transition_epochs,
                    ))
            else:
                set_prepbn_progress(
                    backbone.module, global_step, prepbn_decay_steps)
            capture_simple_gate_stats = (
                simple_gate_enabled
                and simple_gate_stats_interval > 0
                and global_step % simple_gate_stats_interval == 0
            )
            capture_nf_stats = (
                nf_enabled and nf_stats_interval > 0
                and global_step % nf_stats_interval == 0)
            if nf_enabled:
                backbone.module.set_nf_range_tracking(
                    nf_range_loss_weight > 0.0 or capture_nf_stats)
            set_simple_gate_instrumentation(
                backbone.module,
                capture_simple_gate_stats,
                gradient_scale=float(amp.get_scale()) if cfg.fp16 else 1.0,
            )
            layerwise_training_phase = None
            active_layerwise_group = ()
            effective_layerwise_backbone_lr_scale = 1.0
            if herpn_group_schedule:
                fractional_epoch = epoch + step_in_epoch / max(
                    scheduled_steps_per_epoch, 1)
                if (layerwise_poly_enabled
                        and layerwise_poly_strict_recalibrate_before_blend):
                    crossed_group_indices = fractional_group_starts_crossed(
                        epoch,
                        fractional_epoch,
                        herpn_group_epochs,
                        strictly_revalidated_layerwise_groups,
                    )
                    for group_index in crossed_group_indices:
                        strictly_revalidate_layerwise_group_before_blend(
                            group_index)
                backbone.module.set_herpn_blends(herpn_group_blends_at_epoch(
                    fractional_epoch, herpn_conversion_groups,
                    herpn_group_epochs, herpn_transition_epochs))
                if layerwise_poly_staged_training:
                    phase, active_group_index = (
                        layerwise_poly_group_phase_at_epoch(
                            fractional_epoch,
                            layerwise_poly_training_groups,
                            layerwise_poly_training_group_epochs,
                            herpn_transition_epochs,
                        )
                    )
                    layerwise_training_phase = phase
                    if active_group_index is not None:
                        active_layerwise_group = layerwise_poly_training_groups[
                            active_group_index]
                    if phase == "blend":
                        effective_layerwise_backbone_lr_scale = (
                            layerwise_poly_blend_backbone_lr_scale)
                    elif phase == "final_finetune":
                        effective_layerwise_backbone_lr_scale = (
                            layerwise_poly_final_backbone_lr_scale)
                    elif phase == "local_fit":
                        effective_layerwise_backbone_lr_scale = (
                            layerwise_poly_local_fit_backbone_lr_scale)
            elif herpn_enabled and herpn_stage_epochs:
                fractional_epoch = epoch + step_in_epoch / max(
                    scheduled_steps_per_epoch, 1)
                backbone.module.set_herpn_progress(herpn_progress_at_epoch(
                    fractional_epoch, herpn_stage_epochs, herpn_transition_epochs))
            effective_simple_gate_lr_scale = 1.0
            if simple_gate_schedule:
                fractional_epoch = epoch + step_in_epoch / max(
                    scheduled_steps_per_epoch, 1)
                current_gate_blends = simple_gate_blends_at_epoch(
                    fractional_epoch, simple_gate_group_epochs,
                    simple_gate_transition_epochs)
                set_simple_gate_blends(
                    backbone.module,
                    current_gate_blends)
                if simple_gate_current_group_auxiliary:
                    current_group = next(
                        (index for index, blend in enumerate(
                            current_gate_blends) if blend < 1.0),
                        None,
                    )
                    backbone.module.set_simple_gate_auxiliary_groups(
                        () if current_group is None else (current_group,))
                if fractional_epoch >= simple_gate_group_epochs[0]:
                    effective_simple_gate_lr_scale = (
                        simple_gate_lr_multiplier)
            replay_layerwise_tails = (
                layerwise_tail_replay_loader is not None
                and (
                    layerwise_training_phase == "local_fit"
                    or (
                        layerwise_poly_require_full_containment
                        and layerwise_training_phase in (
                            "blend", "final_finetune")
                    )
                )
            )
            if fixed_tail_replay_loader is not None:
                replay_batch = next_fixed_tail_replay_batch()
                replay_img, replay_labels = replay_batch[:2]
                replay_count = min(
                    int(img.shape[0]), int(replay_img.shape[0]))
                if replay_count > 0:
                    img = img.clone()
                    local_labels = local_labels.clone()
                    img[:replay_count].copy_(replay_img[:replay_count].to(
                        device=img.device, dtype=img.dtype,
                        non_blocking=True))
                    local_labels[:replay_count].copy_(
                        replay_labels[:replay_count].to(
                            device=local_labels.device,
                            dtype=local_labels.dtype,
                            non_blocking=True))
            if replay_layerwise_tails:
                replay_batch = next_layerwise_tail_replay_batch()
                replay_img, replay_labels = replay_batch[:2]
                replay_count = min(
                    int(img.shape[0]), int(replay_img.shape[0]))
                if replay_count > 0:
                    # Keep the distributed batch shape fixed for PartialFC and
                    # replace only a prefix with the globally identified tail
                    # samples. Their ordinary stochastic transform is rerun on
                    # every replay visit.
                    img = img.clone()
                    local_labels = local_labels.clone()
                    img[:replay_count].copy_(replay_img[:replay_count].to(
                        device=img.device, dtype=img.dtype,
                        non_blocking=True))
                    local_labels[:replay_count].copy_(
                        replay_labels[:replay_count].to(
                            device=local_labels.device,
                            dtype=local_labels.dtype,
                            non_blocking=True))
            batchnorm_snapshot = (
                snapshot_batchnorm_running_stats(backbone.module)
                if (max_nonfinite_embedding_skips > 0
                    or max_nonfinite_loss_skips > 0)
                else None)
            preserve_batchnorm_snapshot = None
            preserve_all_batchnorm = (
                (layerwise_training_phase == "local_fit"
                 and layerwise_poly_preserve_batchnorm_during_local_fit)
                or (layerwise_training_phase == "blend"
                    and layerwise_poly_preserve_batchnorm_during_blend)
                or (layerwise_training_phase == "final_finetune"
                    and layerwise_poly_preserve_batchnorm_during_final_finetune)
            )
            if preserve_all_batchnorm and not freeze_batchnorm_running_stats:
                preserve_batchnorm_snapshot = (
                    snapshot_batchnorm_running_stats(backbone.module))
            if freeze_batchnorm_running_stats:
                # Verification restores the whole backbone to train mode.
                # Reassert the frozen-stat policy before every forward.
                freeze_batchnorm_for_training(
                    backbone.module, affine=freeze_batchnorm_affine)
            backbone_output = backbone(img)
            if preserve_batchnorm_snapshot is not None:
                # Use train-mode batch moments to keep rare-tail forwards
                # stable, but do not let their mutable running buffers drift
                # the accepted inference graph.
                restore_batchnorm_running_stats(preserve_batchnorm_snapshot)
            if cryptoface_patch_training:
                local_embeddings, patch_pred, patch_target = backbone_output
            else:
                local_embeddings = backbone_output
            embeddings_finite = torch.isfinite(local_embeddings).all()
            finite_rank_count = embeddings_finite.to(dtype=torch.long)
            if distributed.is_available() and distributed.is_initialized():
                distributed.all_reduce(
                    finite_rank_count, op=distributed.ReduceOp.SUM)
            finite_rank_count = int(finite_rank_count.item())
            if finite_rank_count != world_size:
                nonfinite_embedding_skips += 1
                gate_context = ""
                if last_simple_gate_snapshot:
                    worst_name, worst_stats = max(
                        last_simple_gate_snapshot.items(),
                        key=lambda item: item[1]["product_absmax"])
                    gate_context = (
                        f"; last_gate_profile_step="
                        f"{last_simple_gate_snapshot_step}, "
                        f"worst_gate={worst_name}, "
                        f"product_absmax={worst_stats['product_absmax']:.6g}, "
                        f"product_p999={worst_stats['product_p999']:.6g}, "
                        f"blend={worst_stats.get('blend', float('nan')):.3f}")
                if last_nf_snapshot:
                    worst_name, worst_stats = max(
                        last_nf_snapshot.items(),
                        key=lambda item: item[1]["product_absmax"])
                    gate_context += (
                        f"; last_nf_profile_step={last_nf_snapshot_step}, "
                        f"worst_nf_block={worst_name}, "
                        f"product_absmax={worst_stats['product_absmax']:.6g}, "
                        f"product_p999={worst_stats['product_p999']:.6g}")
                if (max_nonfinite_embedding_skips <= 0
                        or nonfinite_embedding_skips
                        > max_nonfinite_embedding_skips):
                    raise FloatingPointError(
                        f"Non-finite embeddings at global_step={global_step}; "
                        f"finite_ranks={finite_rank_count}/{world_size}, "
                        f"skip_count={nonfinite_embedding_skips}/"
                        f"{max_nonfinite_embedding_skips}{gate_context}")
                if rank == 0:
                    logging.warning(
                        "Skipping synchronized training batch at "
                        "global_step=%d because embeddings were finite on "
                        "%d/%d ranks; skip_count=%d/%d%s",
                        global_step, finite_rank_count, world_size,
                        nonfinite_embedding_skips,
                        max_nonfinite_embedding_skips, gate_context)
                opt.zero_grad()
                if batchnorm_snapshot is not None:
                    restore_batchnorm_running_stats(batchnorm_snapshot)
                # The auxiliary SimpleGate tensors retain the current
                # autograd graph. A skipped batch has no backward pass to
                # release it, so clear those references before the next
                # forward or two full graphs can overlap and exhaust VRAM.
                clear_gate_cache = getattr(
                    backbone.module,
                    "clear_simple_gate_cached_tensors",
                    None,
                )
                if clear_gate_cache is not None:
                    clear_gate_cache()
                clear_nf_cache = getattr(
                    backbone.module, "clear_nf_cached_tensors", None)
                if clear_nf_cache is not None:
                    clear_nf_cache()
                clear_frozen_std_cache = getattr(
                    backbone.module, "clear_frozen_std_cached_tensors", None)
                if clear_frozen_std_cache is not None:
                    clear_frozen_std_cache()
                del backbone_output, local_embeddings
                torch.cuda.empty_cache()
                if not lr_scheduler_step_per_epoch:
                    lr_scheduler.step()
                set_simple_gate_instrumentation(backbone.module, False)
                continue
            embedding_distillation_loss = local_embeddings.new_zeros(())
            if embedding_teacher is not None:
                # Run the no-grad teacher before PartialFC constructs its
                # large classifier graph, reducing peak memory on V100s.
                with torch.no_grad():
                    teacher_embeddings = embedding_teacher(img)
                if teacher_embeddings.shape != local_embeddings.shape:
                    raise ValueError(
                        "Embedding teacher/student shape mismatch: "
                        f"teacher={tuple(teacher_embeddings.shape)}, "
                        f"student={tuple(local_embeddings.shape)}")
                if not torch.isfinite(teacher_embeddings).all():
                    raise FloatingPointError(
                        "Non-finite teacher embeddings at "
                        f"global_step={global_step}")
                student_unit = F.normalize(
                    local_embeddings.float(), dim=1, eps=1e-6)
                teacher_unit = F.normalize(
                    teacher_embeddings.float(), dim=1, eps=1e-6)
                embedding_distillation_loss = 0.5 * (
                    student_unit - teacher_unit).square().sum(dim=1).mean()
                if not torch.isfinite(embedding_distillation_loss):
                    raise FloatingPointError(
                        "Non-finite embedding distillation loss at "
                        f"global_step={global_step}")
            if cryptoface_patch_training:
                local_labels = local_labels.squeeze().long()
                logits = module_partial_fc(local_embeddings, local_labels)
                loss: torch.Tensor = criterion(logits, local_labels)
                loss_jigsaw = F.cross_entropy(patch_pred, patch_target)
                loss = loss + float(getattr(cfg, "patch_cnn_jigsaw_weight", 0.005)) * loss_jigsaw
            elif task_loss_weight == 0.0:
                # Range-only/distillation recovery does not need to construct
                # PartialFC's large classifier graph.  Keep a connected scalar
                # so auxiliary losses can be added normally below.
                loss = local_embeddings.flatten()[0] * 0.0
            else:
                loss: torch.Tensor = module_partial_fc(local_embeddings, local_labels)
            task_loss = loss
            if task_loss_weight != 1.0:
                loss = loss * task_loss_weight
            if pillar_enabled and pillar_task_loss_weight != 1.0:
                loss = loss * pillar_task_loss_weight
            range_penalty = local_embeddings.new_zeros(())
            distillation_loss = local_embeddings.new_zeros(())
            simple_gate_range_penalty = local_embeddings.new_zeros(())
            simple_gate_distillation_loss = local_embeddings.new_zeros(())
            nf_range_penalty = local_embeddings.new_zeros(())
            precise_relu_range_penalty = local_embeddings.new_zeros(())
            pillar_range_penalty = local_embeddings.new_zeros(())
            frozen_std_auxiliary_loss = local_embeddings.new_zeros(())
            if embedding_teacher is not None:
                loss = loss + (
                    embedding_distill_weight
                    * embedding_distillation_loss)
            if nf_enabled and nf_range_loss_weight > 0.0:
                nf_range_penalty = backbone.module.nf_range_penalty()
                if not torch.isfinite(nf_range_penalty):
                    raise FloatingPointError(
                        "Non-finite NF range penalty at "
                        f"global_step={global_step}")
                loss = loss + nf_range_loss_weight * nf_range_penalty
            conditioning_range_loss = (
                layerwise_training_phase == "local_fit"
                and layerwise_poly_conditioning_backbone_lr_scale > 0.0
                and layerwise_poly_conditioning_range_loss_weight > 0.0
            )
            containment_guard_loss = (
                layerwise_poly_require_full_containment
                and layerwise_training_phase in ("blend", "final_finetune")
                and layerwise_poly_conditioning_range_loss_weight > 0.0
            )
            layerwise_guard_loss = (
                conditioning_range_loss or containment_guard_loss)
            if (herpn_enabled
                    and (herpn_range_loss_weight > 0
                         or layerwise_guard_loss)):
                range_penalty_names = (
                    active_layerwise_group
                    if conditioning_range_loss
                    else (herpn_range_loss_names or None)
                )
                if (layerwise_guard_loss
                        and layerwise_poly_require_full_containment):
                    model_order = tuple(
                        backbone.module.layerwise_poly_activation_names())
                    activations = dict(
                        backbone.module.named_progressive_activations())
                    calibrated_names = tuple(
                        name for name in model_order
                        if activations[name]._scale_is_calibrated)
                    range_penalty_names = calibrated_conversion_prefix(
                        model_order,
                        calibrated_names,
                        layerwise_poly_training_groups,
                    )
                    if not range_penalty_names:
                        raise RuntimeError(
                            "Full-containment training has no calibrated "
                            "activation prefix to guard")
                range_penalty = (
                    backbone.module.herpn_range_penalty(range_penalty_names)
                    if range_penalty_names is not None
                    else backbone.module.herpn_range_penalty()
                )
                if not torch.isfinite(range_penalty):
                    raise FloatingPointError(
                        f"Non-finite HerPN range penalty at global_step={global_step}"
                    )
                effective_range_loss_weight = (
                    layerwise_poly_conditioning_range_loss_weight
                    if layerwise_guard_loss
                    else herpn_range_loss_weight
                )
                loss = loss + effective_range_loss_weight * range_penalty
            if (precise_relu_enabled
                    and precise_relu_range_loss_weight > 0.0):
                precise_relu_range_penalty = (
                    backbone.module.polynomial_range_penalty())
                if not torch.isfinite(precise_relu_range_penalty):
                    raise FloatingPointError(
                        "Non-finite PreciseReLU range penalty at "
                        f"global_step={global_step}")
                loss = loss + (
                    precise_relu_range_loss_weight
                    * precise_relu_range_penalty)
            if pillar_enabled and pillar_effective_coefficient > 0.0:
                pillar_range_penalty = (
                    backbone.module.pillar_range_penalty())
                if not torch.isfinite(pillar_range_penalty):
                    pillar_summary = backbone.module.pillar_range_summary()
                    raise FloatingPointError(
                        "Non-finite PILLAR range penalty at "
                        f"global_step={global_step}; "
                        "input_absmax="
                        f"{float(pillar_summary['input_absmax'].item()):.7g}; "
                        "approximation_outside="
                        f"{float(pillar_summary['approximation_outside_fraction'].item()):.7g}; "
                        "regularization_outside="
                        f"{float(pillar_summary['regularization_outside_fraction'].item()):.7g}")
                loss = loss + (
                    pillar_effective_coefficient
                    * pillar_range_penalty)
                if (rank == 0 and pillar_log_interval > 0
                        and global_step % pillar_log_interval == 0):
                    pillar_summary = backbone.module.pillar_range_summary()
                    logging.info(
                        "PILLAR range step=%d penalty=%.7g weighted=%.7g "
                        "input_absmax=%.7g approximation_outside=%.7g "
                        "regularization_outside=%.7g",
                        global_step,
                        float(pillar_range_penalty.item()),
                        float((pillar_effective_coefficient
                               * pillar_range_penalty).item()),
                        float(pillar_summary["input_absmax"].item()),
                        float(pillar_summary[
                            "approximation_outside_fraction"].item()),
                        float(pillar_summary[
                            "regularization_outside_fraction"].item()),
                    )
            if herpn_enabled and herpn_distill_loss_weight > 0:
                distillation_names = (
                    active_layerwise_group
                    if layerwise_training_phase == "local_fit"
                    else None
                )
                distillation_loss = (
                    backbone.module.herpn_distillation_loss(distillation_names)
                    if distillation_names is not None
                    else backbone.module.herpn_distillation_loss()
                )
                if not torch.isfinite(distillation_loss):
                    raise FloatingPointError(
                        f"Non-finite HerPN distillation loss at global_step={global_step}"
                    )
                loss = loss + herpn_distill_loss_weight * distillation_loss
            if simple_gate_progressive and simple_gate_range_loss_weight > 0:
                simple_gate_range_penalty = (
                    backbone.module.simple_gate_range_penalty())
                if not torch.isfinite(simple_gate_range_penalty):
                    raise FloatingPointError(
                        "Non-finite SimpleGate range penalty at "
                        f"global_step={global_step}")
                loss = loss + (
                    simple_gate_range_loss_weight * simple_gate_range_penalty)
            if simple_gate_progressive and simple_gate_distill_loss_weight > 0:
                simple_gate_distillation_loss = (
                    backbone.module.simple_gate_distillation_loss())
                if not torch.isfinite(simple_gate_distillation_loss):
                    raise FloatingPointError(
                        "Non-finite SimpleGate distillation loss at "
                        f"global_step={global_step}")
                loss = loss + (
                    simple_gate_distill_loss_weight
                    * simple_gate_distillation_loss)
            if frozen_std_enabled and frozen_std_aux_loss_weight > 0.0:
                frozen_std_auxiliary_loss = (
                    backbone.module.frozen_std_auxiliary_loss())
                if not torch.isfinite(frozen_std_auxiliary_loss):
                    raise FloatingPointError(
                        "Non-finite frozen-std auxiliary loss at "
                        f"global_step={global_step}")
                loss = loss + (
                    frozen_std_aux_loss_weight
                    * frozen_std_auxiliary_loss)
            loss_finite = torch.isfinite(loss)
            if max_nonfinite_loss_skips <= 0:
                if not bool(loss_finite.item()):
                    raise FloatingPointError(
                        "Non-finite loss at global_step="
                        f"{global_step}: {loss.item()}")
                finite_loss_rank_count = world_size
            else:
                finite_loss_rank_count = loss_finite.to(dtype=torch.long)
                if (distributed.is_available()
                        and distributed.is_initialized()):
                    distributed.all_reduce(
                        finite_loss_rank_count, op=distributed.ReduceOp.SUM)
                finite_loss_rank_count = int(finite_loss_rank_count.item())
            if (max_nonfinite_loss_skips > 0
                    and finite_loss_rank_count != world_size):
                nonfinite_loss_skips += 1
                if nonfinite_loss_skips > max_nonfinite_loss_skips:
                    raise FloatingPointError(
                        "Non-finite loss at global_step="
                        f"{global_step}; finite_ranks="
                        f"{finite_loss_rank_count}/{world_size}; "
                        f"skip_count={nonfinite_loss_skips}/"
                        f"{max_nonfinite_loss_skips}")
                if rank == 0:
                    logging.warning(
                        "Skipping synchronized training batch at "
                        "global_step=%d because losses were finite on "
                        "%d/%d ranks; skip_count=%d/%d",
                        global_step, finite_loss_rank_count, world_size,
                        nonfinite_loss_skips, max_nonfinite_loss_skips)
                opt.zero_grad()
                if batchnorm_snapshot is not None:
                    restore_batchnorm_running_stats(batchnorm_snapshot)
                clear_gate_cache = getattr(
                    backbone.module, "clear_simple_gate_cached_tensors", None)
                if clear_gate_cache is not None:
                    clear_gate_cache()
                clear_nf_cache = getattr(
                    backbone.module, "clear_nf_cached_tensors", None)
                if clear_nf_cache is not None:
                    clear_nf_cache()
                clear_frozen_std_cache = getattr(
                    backbone.module,
                    "clear_frozen_std_cached_tensors",
                    None,
                )
                if clear_frozen_std_cache is not None:
                    clear_frozen_std_cache()
                del backbone_output, local_embeddings, loss
                torch.cuda.empty_cache()
                if not lr_scheduler_step_per_epoch:
                    lr_scheduler.step()
                set_simple_gate_instrumentation(backbone.module, False)
                continue

            backward_loss = loss
            nonfinite_gradient_step_skipped = False
            if (cfg.gradient_acc > 1
                    and getattr(cfg, "normalize_gradient_accumulation", False)):
                backward_loss = loss / cfg.gradient_acc
            if cfg.fp16:
                amp.scale(backward_loss).backward()
                if global_step % cfg.gradient_acc == 0:
                    amp.unscale_(opt)
                    if layerwise_training_phase == "local_fit":
                        if layerwise_poly_conditioning_backbone_lr_scale > 0.0:
                            retain_layerwise_poly_conditioning_gradients(
                                backbone.module, active_layerwise_group)
                        elif layerwise_poly_freeze_backbone_during_local_fit:
                            retain_only_layerwise_poly_group_gradients(
                                backbone.module, active_layerwise_group)
                    named_gradient_modules = (
                        ("backbone", backbone),
                        ("partial_fc", module_partial_fc),
                    )
                    local_nonfinite_tensor_count = torch.zeros(
                        (), device=local_embeddings.device, dtype=torch.long)
                    gradient_diagnostics = ()
                    if skip_nonfinite_gradients:
                        local_nonfinite_tensor_count = (
                            nonfinite_gradient_tensor_count(
                                named_gradient_modules,
                                device=local_embeddings.device))
                    elif getattr(cfg, "check_finite_grads", False):
                        check_finite_gradients(backbone, "backbone", global_step)
                        check_finite_gradients(module_partial_fc, "partial_fc", global_step)
                    if getattr(cfg, "gradient_clip_type", "norm") == "value":
                        torch.nn.utils.clip_grad_value_(clipped_params, grad_clip)
                        total_norm = torch.tensor(0.0, device=local_embeddings.device)
                    elif getattr(cfg, "stable_gradient_clip", False):
                        total_norm = clip_grad_norm_stable(
                            clipped_params, grad_clip,
                            error_if_nonfinite=False)
                    else:
                        total_norm = torch.nn.utils.clip_grad_norm_(
                            clipped_params, grad_clip, error_if_nonfinite=False
                        )
                    local_bad_gradient = (
                        local_nonfinite_tensor_count > 0
                    ) | (~torch.isfinite(total_norm).to(
                        device=local_embeddings.device))
                    bad_gradient_rank = (
                        local_bad_gradient.to(dtype=torch.long) * (rank + 1))
                    if (distributed.is_available()
                            and distributed.is_initialized()):
                        distributed.all_reduce(
                            bad_gradient_rank,
                            op=distributed.ReduceOp.MAX)
                    bad_gradient_rank_code = int(bad_gradient_rank.item())
                    global_bad_gradient = bool(bad_gradient_rank_code)
                    source_rank = bad_gradient_rank_code - 1
                    if not global_bad_gradient:
                        with temporary_optimizer_lr_scale(
                                opt, effective_simple_gate_lr_scale):
                            if layerwise_poly_staged_training:
                                with temporary_optimizer_lr_scale(
                                        opt,
                                        effective_layerwise_backbone_lr_scale,
                                        scope="backbone"):
                                    with temporary_optimizer_lr_scale(
                                            opt,
                                            layerwise_poly_optimizer_lr_scale,
                                            scope="layerwise_poly"):
                                        amp.step(opt)
                            else:
                                amp.step(opt)
                        amp.update()
                    elif skip_nonfinite_gradients:
                        nonfinite_gradient_step_skipped = True
                        nonfinite_gradient_skips += 1
                        local_bad_gradient = bool(local_bad_gradient.item())
                        local_nonfinite_tensors = int(
                            local_nonfinite_tensor_count.item())
                        if local_nonfinite_tensors:
                            _, gradient_diagnostics = (
                                nonfinite_gradient_diagnostics(
                                    named_gradient_modules))
                        current_scale = torch.tensor(
                            float(amp.get_scale()),
                            device=local_embeddings.device,
                            dtype=torch.float64,
                        )
                        if (distributed.is_available()
                                and distributed.is_initialized()):
                            distributed.all_reduce(
                                current_scale, op=distributed.ReduceOp.MIN)
                        new_scale = max(
                            float(current_scale.item())
                            * nonfinite_gradient_scale_backoff,
                            min_amp_scale,
                        )
                        local_details = "; ".join(
                            f"{item['name']} shape={item['shape']} "
                            f"nonfinite={item['nonfinite_elements']} "
                            f"finite_absmax={item['finite_absmax']:.6g}"
                            for item in gradient_diagnostics
                        )
                        if rank == 0:
                            logging.warning(
                                "Skipping synchronized FP16 optimizer step "
                                "at global_step=%d: source_rank=%d, "
                                "nonfinite_tensors_on_rank0=%d, grad_norm=%s, "
                                "amp_scale=%.7g->%.7g, skip_count=%d%s%s",
                                global_step, source_rank,
                                local_nonfinite_tensors,
                                str(float(total_norm.item())),
                                float(current_scale.item()), new_scale,
                                nonfinite_gradient_skips,
                                "; " if local_details else "",
                                local_details,
                            )
                        elif local_bad_gradient:
                            logging.warning(
                                "Rank %d detected non-finite FP16 gradients "
                                "at global_step=%d; optimizer step skipped%s%s",
                                rank, global_step,
                                "; " if local_details else "",
                                local_details,
                            )
                        amp.update(new_scale=new_scale)
                        if (max_nonfinite_gradient_skips > 0
                                and nonfinite_gradient_skips
                                > max_nonfinite_gradient_skips):
                            raise FloatingPointError(
                                "Exceeded max_nonfinite_gradient_skips="
                                f"{max_nonfinite_gradient_skips} at "
                                f"global_step={global_step}")
                    else:
                        raise FloatingPointError(
                            "Non-finite FP16 gradient norm at "
                            f"global_step={global_step}: "
                            f"{float(total_norm.item())}")
                    opt.zero_grad()
                    if (nonfinite_gradient_step_skipped
                            and batchnorm_snapshot is not None):
                        restore_batchnorm_running_stats(batchnorm_snapshot)
            else:
                backward_loss.backward()
                if global_step % cfg.gradient_acc == 0:
                    if layerwise_training_phase == "local_fit":
                        if layerwise_poly_conditioning_backbone_lr_scale > 0.0:
                            retain_layerwise_poly_conditioning_gradients(
                                backbone.module, active_layerwise_group)
                        elif layerwise_poly_freeze_backbone_during_local_fit:
                            retain_only_layerwise_poly_group_gradients(
                                backbone.module, active_layerwise_group)
                    named_gradient_modules = (
                        ("backbone", backbone),
                        ("partial_fc", module_partial_fc),
                    )
                    local_nonfinite_tensor_count = torch.zeros(
                        (), device=local_embeddings.device, dtype=torch.long)
                    if skip_nonfinite_gradients:
                        local_nonfinite_tensor_count = (
                            nonfinite_gradient_tensor_count(
                                named_gradient_modules,
                                device=local_embeddings.device))
                    else:
                        check_finite_gradients(
                            backbone, "backbone", global_step)
                        check_finite_gradients(
                            module_partial_fc, "partial_fc", global_step)
                    if getattr(cfg, "gradient_clip_type", "norm") == "value":
                        torch.nn.utils.clip_grad_value_(clipped_params, grad_clip)
                        total_norm = torch.tensor(
                            0.0, device=local_embeddings.device)
                    elif getattr(cfg, "stable_gradient_clip", False):
                        total_norm = clip_grad_norm_stable(
                            clipped_params, grad_clip,
                            error_if_nonfinite=not skip_nonfinite_gradients)
                    else:
                        total_norm = torch.nn.utils.clip_grad_norm_(
                            clipped_params, grad_clip,
                            error_if_nonfinite=not skip_nonfinite_gradients)
                    local_bad_gradient = (
                        local_nonfinite_tensor_count > 0
                    ) | (~torch.isfinite(total_norm).to(
                        device=local_embeddings.device))
                    bad_gradient_rank = (
                        local_bad_gradient.to(dtype=torch.long) * (rank + 1))
                    if (distributed.is_available()
                            and distributed.is_initialized()):
                        distributed.all_reduce(
                            bad_gradient_rank,
                            op=distributed.ReduceOp.MAX)
                    bad_gradient_rank_code = int(bad_gradient_rank.item())
                    global_bad_gradient = bool(bad_gradient_rank_code)
                    source_rank = bad_gradient_rank_code - 1
                    gradient_norm_warning_threshold = float(getattr(
                        cfg, "gradient_norm_warning_threshold", 100.0))
                    gradient_norm_warning_interval = max(1, int(getattr(
                        cfg, "gradient_norm_warning_interval", 100)))
                    if (rank == 0
                            and gradient_norm_warning_threshold > 0.0
                            and global_step % gradient_norm_warning_interval == 0
                            and total_norm.item()
                            > gradient_norm_warning_threshold):
                        logging.warning(
                            "Large finite pre-clip gradient norm at "
                            "global_step=%d: %.6g (clipped to %.6g)",
                            global_step, total_norm.item(), grad_clip)
                    if not global_bad_gradient:
                        with temporary_optimizer_lr_scale(
                                opt, effective_simple_gate_lr_scale):
                            if layerwise_poly_staged_training:
                                with temporary_optimizer_lr_scale(
                                        opt,
                                        effective_layerwise_backbone_lr_scale,
                                        scope="backbone"):
                                    with temporary_optimizer_lr_scale(
                                            opt,
                                            layerwise_poly_optimizer_lr_scale,
                                            scope="layerwise_poly"):
                                        opt.step()
                            else:
                                opt.step()
                    elif skip_nonfinite_gradients:
                        nonfinite_gradient_step_skipped = True
                        nonfinite_gradient_skips += 1
                        local_bad_gradient = bool(local_bad_gradient.item())
                        local_nonfinite_tensors = int(
                            local_nonfinite_tensor_count.item())
                        gradient_diagnostics = ()
                        if local_nonfinite_tensors:
                            _, gradient_diagnostics = (
                                nonfinite_gradient_diagnostics(
                                    named_gradient_modules))
                        local_details = "; ".join(
                            f"{item['name']} shape={item['shape']} "
                            f"nonfinite={item['nonfinite_elements']} "
                            f"finite_absmax={item['finite_absmax']:.6g}"
                            for item in gradient_diagnostics
                        )
                        if rank == 0:
                            logging.warning(
                                "Skipping synchronized FP32 optimizer step "
                                "at global_step=%d: source_rank=%d, "
                                "nonfinite_tensors_on_rank0=%d, grad_norm=%s, "
                                "skip_count=%d%s%s",
                                global_step, source_rank,
                                local_nonfinite_tensors,
                                str(float(total_norm.item())),
                                nonfinite_gradient_skips,
                                "; " if local_details else "",
                                local_details,
                            )
                        elif local_bad_gradient:
                            logging.warning(
                                "Rank %d detected non-finite FP32 gradients "
                                "at global_step=%d; optimizer step skipped%s%s",
                                rank, global_step,
                                "; " if local_details else "",
                                local_details,
                            )
                        if (max_nonfinite_gradient_skips > 0
                                and nonfinite_gradient_skips
                                > max_nonfinite_gradient_skips):
                            raise FloatingPointError(
                                "Exceeded max_nonfinite_gradient_skips="
                                f"{max_nonfinite_gradient_skips} at "
                                f"global_step={global_step}")
                        if batchnorm_snapshot is not None:
                            restore_batchnorm_running_stats(
                                batchnorm_snapshot)
                    else:
                        raise FloatingPointError(
                            "Non-finite FP32 gradient norm at "
                            f"global_step={global_step}: "
                            f"{float(total_norm.item())}")
                    opt.zero_grad()
            del batchnorm_snapshot
            if (not lr_scheduler_step_per_epoch
                    and not nonfinite_gradient_step_skipped):
                lr_scheduler.step()

            with torch.no_grad():
                if capture_simple_gate_stats:
                    gate_stats = collect_simple_gate_stats(backbone.module)
                    last_simple_gate_snapshot = log_simple_gate_stats(
                        gate_stats,
                        global_step,
                        summary_writer=summary_writer,
                        wandb_logger=wandb_logger,
                    )
                    last_simple_gate_snapshot_step = global_step
                    set_simple_gate_instrumentation(backbone.module, False)
                if capture_nf_stats:
                    last_nf_snapshot = log_nf_range_stats(
                        backbone.module.nf_range_summary(),
                        global_step,
                        summary_writer=summary_writer,
                        wandb_logger=wandb_logger,
                    )
                    last_nf_snapshot_step = global_step
                if wandb_logger:
                    wandb_logger.log({
                        'Loss/Task Loss': task_loss.item(),
                        'Loss/Total Loss': loss.item(),
                        'Loss/Step Loss': loss.item(),
                        'Loss/Train Loss': loss_am.avg,
                        'Loss/HerPN Range Penalty': range_penalty.item(),
                        'Loss/HerPN Distillation': distillation_loss.item(),
                        'Loss/SimpleGate Range Penalty': (
                            simple_gate_range_penalty.item()),
                        'Loss/SimpleGate Distillation': (
                            simple_gate_distillation_loss.item()),
                        'Loss/NF Range Penalty': nf_range_penalty.item(),
                        'Loss/PreciseReLU Range Penalty': (
                            precise_relu_range_penalty.item()),
                        'Loss/PILLAR Range Penalty': (
                            pillar_range_penalty.item()),
                        'Loss/Embedding Distillation': (
                            embedding_distillation_loss.item()),
                        'Loss/Frozen Std Auxiliary': (
                            frozen_std_auxiliary_loss.item()),
                        'Process/SimpleGate Progress': (
                            sum(simple_gate_blends_at_epoch(
                                epoch + step_in_epoch / max(
                                    scheduled_steps_per_epoch, 1),
                                simple_gate_group_epochs,
                                simple_gate_transition_epochs))
                            if simple_gate_schedule else 0.0),
                        'Process/HerPN Progress': (
                            float(backbone.module.herpn_progress.item())
                            if herpn_enabled else 0.0),
                        'Process/PreciseReLU Progress': (
                            float(backbone.module.polynomial_progress.item())
                            if precise_relu_enabled else 0.0),
                        'Process/PILLAR Beta': pillar_effective_coefficient,
                        'Process/PILLAR Gamma': pillar_effective_exponent,
                        'Process/Step': global_step,
                        'Process/Epoch': epoch
                    })

                if (summary_writer is not None
                        and global_step % cfg.frequent == 0):
                    summary_writer.add_scalar(
                        'Loss/NF Range Penalty',
                        nf_range_penalty.item(), global_step)
                    summary_writer.add_scalar(
                        'Loss/PreciseReLU Range Penalty',
                        precise_relu_range_penalty.item(), global_step)
                    summary_writer.add_scalar(
                        'Loss/PILLAR Range Penalty',
                        pillar_range_penalty.item(), global_step)
                    summary_writer.add_scalar(
                        'Loss/Embedding Distillation',
                        embedding_distillation_loss.item(), global_step)
                    summary_writer.add_scalar(
                        'Loss/Frozen Std Auxiliary',
                        frozen_std_auxiliary_loss.item(), global_step)

                if (summary_writer is not None and pillar_enabled and
                        global_step % cfg.frequent == 0):
                    pillar_summary = backbone.module.pillar_range_summary()
                    summary_writer.add_scalar(
                        'PILLAR/Input Abs Max',
                        float(pillar_summary['input_absmax'].item()),
                        global_step)
                    summary_writer.add_scalar(
                        'PILLAR/Approximation Outside Fraction',
                        float(pillar_summary[
                            'approximation_outside_fraction'].item()),
                        global_step)
                    summary_writer.add_scalar(
                        'PILLAR/Regularization Outside Fraction',
                        float(pillar_summary[
                            'regularization_outside_fraction'].item()),
                        global_step)
                    summary_writer.add_scalar(
                        'Process/PILLAR Beta',
                        pillar_effective_coefficient, global_step)
                    summary_writer.add_scalar(
                        'Process/PILLAR Gamma',
                        pillar_effective_exponent, global_step)

                if (summary_writer is not None and herpn_enabled and
                        global_step % cfg.frequent == 0):
                    range_summary = backbone.module.herpn_range_summary()
                    summary_writer.add_scalar(
                        'Loss/Task Loss', task_loss.item(), global_step)
                    summary_writer.add_scalar(
                        'Loss/Total Loss', loss.item(), global_step)
                    summary_writer.add_scalar(
                        'Loss/HerPN Range Penalty', range_penalty.item(), global_step)
                    summary_writer.add_scalar(
                        'Loss/HerPN Distillation', distillation_loss.item(), global_step)
                    summary_writer.add_scalar(
                        'Process/HerPN Progress',
                        float(backbone.module.herpn_progress.item()), global_step)
                    summary_writer.add_scalar(
                        'HerPN/Input Abs Max',
                        float(range_summary['input_absmax'].item()), global_step)
                    summary_writer.add_scalar(
                        'HerPN/Outside Range Fraction',
                        float(range_summary['outside_fraction'].item()), global_step)
                    residual_summary_fn = getattr(
                        backbone.module, "residual_scale_summary", None)
                    if residual_summary_fn is not None:
                        residual_summary = residual_summary_fn()
                        for name, value in residual_summary.items():
                            summary_writer.add_scalar(
                                "ResidualScale/" + name.capitalize(),
                                float(value.item()), global_step)
                if (summary_writer is not None and precise_relu_enabled
                        and global_step % cfg.frequent == 0):
                    precise_summary = (
                        backbone.module.polynomial_range_summary())
                    summary_writer.add_scalar(
                        'Process/PreciseReLU Progress',
                        float(backbone.module.polynomial_progress.item()),
                        global_step)
                    summary_writer.add_scalar(
                        'PreciseReLU/Input Abs Max',
                        float(precise_summary['input_absmax'].item()),
                        global_step)
                    summary_writer.add_scalar(
                        'PreciseReLU/Outside Range Fraction',
                        float(precise_summary['outside_fraction'].item()),
                        global_step)
                if (summary_writer is not None and simple_gate_progressive
                        and global_step % cfg.frequent == 0):
                    summary_writer.add_scalar(
                        'Loss/SimpleGate Range Penalty',
                        simple_gate_range_penalty.item(), global_step)
                    summary_writer.add_scalar(
                        'Loss/SimpleGate Distillation',
                        simple_gate_distillation_loss.item(), global_step)
                    if simple_gate_schedule:
                        current_gate_blends = simple_gate_blends_at_epoch(
                            epoch + step_in_epoch / max(
                                scheduled_steps_per_epoch, 1),
                            simple_gate_group_epochs,
                            simple_gate_transition_epochs)
                        for group_index, blend in enumerate(
                                current_gate_blends):
                            summary_writer.add_scalar(
                                f'Process/SimpleGate Group {group_index} Blend',
                                blend, global_step)
                if (summary_writer is not None and prepbn_decay_steps > 0
                        and global_step % cfg.frequent == 0):
                    summary_writer.add_scalar(
                        'RepBatchNorm/Transition Progress',
                        min(global_step / prepbn_decay_steps, 1.0),
                        global_step,
                    )
                if (summary_writer is not None and affine_group_schedule
                        and global_step % cfg.frequent == 0):
                    current_affine_blends = simple_gate_blends_at_epoch(
                        epoch + step_in_epoch / max(
                            scheduled_steps_per_epoch, 1),
                        affine_group_epochs,
                        affine_group_transition_epochs,
                    )
                    summary_writer.add_scalar(
                        "AffineNorm/ConvertedGroups",
                        sum(current_affine_blends),
                        global_step,
                    )
                if nf_enabled:
                    backbone.module.clear_nf_cached_tensors()
                    backbone.module.set_nf_range_tracking(False)
                    
                loss_am.update(loss.item(), 1)
                callback_logging(
                    global_step, loss_am, epoch, cfg.fp16,
                    lr_scheduler.get_last_lr()[0]
                    * effective_simple_gate_lr_scale
                    * effective_layerwise_backbone_lr_scale,
                    amp)

                if global_step % cfg.verbose == 0 and global_step > 0:
                    if rank == 0 and getattr(cfg, "save_validation_snapshots", False):
                        torch.save(
                            backbone.module.state_dict(),
                            os.path.join(cfg.output, "model_validation.pt"),
                        )
                    if (pillar_enabled
                            and epoch < pillar_skip_verification_epochs):
                        if rank == 0:
                            logging.info(
                                "Skipping verification at step %d during "
                                "PILLAR warm-up epoch %d/%d",
                                global_step, epoch + 1,
                                pillar_skip_verification_epochs,
                            )
                    elif (validate_after_prepbn_transition
                            and not prepbn_transition_complete(
                                global_step, prepbn_decay_steps)):
                        if rank == 0:
                            logging.info(
                                "Skipping verification at step %d: RepBatchNorm "
                                "transition is %.2f%% complete",
                                global_step,
                                100.0 * global_step / max(prepbn_decay_steps, 1),
                            )
                    else:
                        if pillar_enabled:
                            strict_pillar_validation = (
                                pillar_validation_is_strict_at_epoch(
                                    epoch,
                                    pillar_strict_verification_epoch))
                            callback_verification.fail_on_nonfinite = (
                                bool(getattr(
                                    cfg, "fail_on_nonfinite_val", False))
                                and strict_pillar_validation)
                            callback_verification.max_embedding_abs = (
                                getattr(
                                    cfg, "max_validation_embedding_abs", None)
                                if strict_pillar_validation else None)
                            if rank == 0 and not strict_pillar_validation:
                                logging.info(
                                    "PILLAR diagnostic verification at step "
                                    "%d (epoch %d); strict finite/range gate "
                                    "starts at epoch %d",
                                    global_step, epoch,
                                    pillar_strict_verification_epoch)
                        callback_verification(global_step, backbone.module)

        if lr_scheduler_step_per_epoch:
            lr_scheduler.step()

        checkpoint_interval = int(getattr(cfg, "checkpoint_interval_epochs", 0))
        if (cfg.save_all_states or
                (checkpoint_interval > 0
                 and (epoch + 1) % checkpoint_interval == 0)):
            checkpoint = {
                "epoch": epoch + 1,
                "global_step": global_step,
                "state_dict_backbone": backbone.module.state_dict(),
                "state_dict_softmax_fc": module_partial_fc.state_dict(),
                "state_optimizer": opt.state_dict(),
                "state_lr_scheduler": lr_scheduler.state_dict(),
                "completed_herpn_groups": completed_herpn_groups,
                "herpn_conversion_groups": herpn_conversion_groups,
                "completed_simple_gate_groups": completed_simple_gate_groups,
                "repbn_gate_recalibrated": repbn_gate_recalibrated,
                "simple_gate_grouping": str(getattr(
                    cfg, "simple_gate_grouping", "stage_chunks")),
                "affine_group_epochs": affine_group_epochs,
                "affine_group_names": (
                    affine_group_names if affine_group_schedule else ()),
                "frozen_std_group_names": frozen_std_group_names,
                "frozen_std_group_steps": frozen_std_steps,
                "frozen_std_transition_steps": frozen_std_transition_steps,
                "completed_frozen_std_groups": completed_frozen_std_groups,
            }
            atomic_torch_save(
                checkpoint,
                os.path.join(cfg.output, f"checkpoint_gpu_{rank}.pt"))

        if rank == 0:
            path_module = os.path.join(cfg.output, "model.pt")
            backbone_state = backbone.module.state_dict()
            atomic_torch_save(backbone_state, path_module)
            epoch_model_interval = int(getattr(cfg, "epoch_model_interval", 0))
            if (getattr(cfg, "save_epoch_models", False)
                    and epoch_model_interval > 0
                    and (epoch + 1) % epoch_model_interval == 0):
                epoch_model_path = os.path.join(
                    cfg.output, f"model_epoch_{epoch + 1:02d}.pt")
                atomic_torch_save(backbone_state, epoch_model_path)
                logging.info(
                    "Saved inference snapshot for epoch %d to %s",
                    epoch + 1, epoch_model_path)

            if wandb_logger and cfg.save_artifacts:
                artifact_name = f"{run_name}_E{epoch}"
                model = wandb.Artifact(artifact_name, type='model')
                model.add_file(path_module)
                wandb_logger.log_artifact(model)
                
        if cfg.dali:
            train_loader.reset()

    # The post-group audit proves the BN-recalibrated boundary, but subsequent
    # hold/final-finetune updates can still move that graph.  Re-scan every
    # accepted interval after the final optimizer step and persist a distinct
    # checkpoint only when this final state remains fully contained.
    if (layerwise_poly_enabled
            and layerwise_poly_require_full_containment
            and completed_herpn_groups > 0):
        final_contained_names = tuple(
            name
            for group in herpn_conversion_groups[:completed_herpn_groups]
            for name in group
        )
        if rank == 0:
            logging.info(
                "Running final complete-domain containment audit for "
                "%d accepted polynomial group(s): %s",
                completed_herpn_groups,
                ", ".join(final_contained_names),
            )
        final_containment_results = calibrate_layerwise_poly_group(
            final_contained_names,
            allow_recalibration=True,
            provisional=False,
        )
        if rank == 0:
            final_audited_path = os.path.join(
                cfg.output,
                "model_herpn_final_containment_audited.pt",
            )
            atomic_torch_save(
                {
                    "state_dict_backbone": backbone.module.state_dict(),
                    "herpn_group": completed_herpn_groups,
                    "herpn_conversion_groups": herpn_conversion_groups,
                    "global_step": global_step,
                    "containment_results": final_containment_results,
                    "scan_both_orientations": bool(
                        layerwise_poly_scan_both_orientations),
                },
                final_audited_path,
            )
            logging.info(
                "Saved final containment-audited HerPN checkpoint to %s",
                final_audited_path,
            )
        if (distributed.is_available()
                and distributed.is_initialized()):
            distributed.barrier()
        callback_verification(global_step, backbone.module)

    if simple_gate_schedule:
        set_simple_gate_blends(
            backbone.module, (1.0,) * len(simple_gate_group_epochs))
    if affine_group_schedule:
        backbone.module.set_affine_group_blends(
            (1.0,) * len(affine_group_epochs))
    if (frozen_std_enabled
            and getattr(cfg, "frozen_std_require_full_conversion", True)
            and completed_frozen_std_groups != len(frozen_std_group_names)):
        raise RuntimeError(
            "Training ended before every frozen-std group was converted: "
            f"{completed_frozen_std_groups}/{len(frozen_std_group_names)}")

    prepbn_bn_stat_epochs = int(getattr(cfg, "prepbn_bn_stat_epochs", 0))
    if prepbn_bn_stat_epochs > 0:
        if rank == 0:
            logging.info("Refreshing PRepBN BatchNorm statistics for %d epoch(s)", prepbn_bn_stat_epochs)
        set_prepbn_progress(backbone.module, prepbn_decay_steps, prepbn_decay_steps)
        recalibration_batches = recalibrate_prepbn_batchnorm(
            backbone,
            train_loader,
            prepbn_bn_stat_epochs,
            cfg.num_epoch,
            dali=cfg.dali,
        )
        if rank == 0:
            logging.info(
                "Refreshed final RepBatchNorm statistics with %d batches",
                recalibration_batches,
            )

    final_gate_profile = profile_simple_gate_ranges(
        backbone,
        train_loader,
        int(getattr(cfg, "simple_gate_final_profile_batches", 0)),
        dali=cfg.dali,
    )
    if final_gate_profile and rank == 0:
        profile_path = os.path.join(cfg.output, "simple_gate_final_profile.json")
        with open(profile_path, "w", encoding="utf-8") as profile_file:
            json.dump(final_gate_profile, profile_file, indent=2, sort_keys=True)
        profile_layers = {
            name: stats for name, stats in final_gate_profile.items()
            if not name.startswith("_")
        }
        worst_name, worst_stats = max(
            profile_layers.items(),
            key=lambda item: item[1]["product_absmax"],
        )
        logging.info(
            "Final SimpleGate profile saved to %s; worst product range is "
            "%s absmax=%.6g p99.9=%.6g outside=%.6g",
            profile_path,
            worst_name,
            worst_stats["product_absmax"],
            worst_stats["product_p999"],
            worst_stats["product_outside_fraction"],
        )

    if (getattr(cfg, "final_verification_after_prepbn", False)
            or getattr(cfg, "final_verification_after_frozen_std", False)):
        if rank == 0:
            logging.info("Running final verification with fully converted normalization")
        callback_verification(global_step, backbone.module)

    if rank == 0:
        path_module = os.path.join(cfg.output, "model.pt")
        atomic_torch_save(backbone.module.state_dict(), path_module)
        
        if wandb_logger and cfg.save_artifacts:
            artifact_name = f"{run_name}_Final"
            model = wandb.Artifact(artifact_name, type='model')
            model.add_file(path_module)
            wandb_logger.log_artifact(model)



if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser(
        description="Distributed Arcface Training in Pytorch")
    parser.add_argument("config", type=str, help="py config file")
    main(parser.parse_args())

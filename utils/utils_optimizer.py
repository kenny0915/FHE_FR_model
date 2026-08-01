"""Optimizer parameter grouping helpers."""

from contextlib import contextmanager

import torch.nn as nn


_NORMALIZATION_TYPES = (
    nn.modules.batchnorm._BatchNorm,
    nn.GroupNorm,
    nn.LayerNorm,
    nn.LocalResponseNorm,
    nn.PReLU,
)


@contextmanager
def temporary_optimizer_lr_scale(optimizer, scale):
    """Apply an LR multiplier to one optimizer step without affecting its scheduler."""
    scale = float(scale)
    if scale <= 0.0:
        raise ValueError("optimizer LR scale must be positive")
    original_lrs = [group["lr"] for group in optimizer.param_groups]
    try:
        for group, original_lr in zip(optimizer.param_groups, original_lrs):
            group["lr"] = original_lr * scale
        yield
    finally:
        for group, original_lr in zip(optimizer.param_groups, original_lrs):
            group["lr"] = original_lr


def split_weight_decay_parameters(module):
    """Split trainable parameters into decay and no-decay groups.

    Polynomial activation coefficients, normalization affine parameters, and
    biases are calibration-sensitive and should not be pulled toward zero.
    Convolution and linear weights retain the configured weight decay.
    """
    no_decay_ids = set()
    for submodule in module.modules():
        exclude_module = (
            isinstance(submodule, _NORMALIZATION_TYPES)
            or submodule.__class__.__name__ == "HerPN"
            or getattr(submodule, "exclude_from_weight_decay", False)
        )
        for name, parameter in submodule.named_parameters(recurse=False):
            if parameter.requires_grad and (exclude_module or name == "bias"):
                no_decay_ids.add(id(parameter))

    decay = []
    no_decay = []
    seen = set()
    for parameter in module.parameters():
        if not parameter.requires_grad:
            continue
        parameter_id = id(parameter)
        if parameter_id in seen:
            continue
        seen.add(parameter_id)
        target = no_decay if parameter_id in no_decay_ids else decay
        target.append(parameter)

    return decay, no_decay

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


def select_gradient_clip_parameters(optimizer, backbone, scope="all"):
    """Select the parameters controlled by the configured gradient clip.

    ``backbone`` matches single-network recipes such as NAFNet without letting
    a large PartialFC classifier dominate the shared gradient norm.
    """
    if scope == "all":
        parameters = [
            parameter
            for group in optimizer.param_groups
            for parameter in group["params"]
            if parameter.requires_grad
        ]
    elif scope == "backbone":
        parameters = [
            parameter for parameter in backbone.parameters()
            if parameter.requires_grad
        ]
    else:
        raise ValueError(
            "gradient_clip_scope must be 'all' or 'backbone', got "
            f"{scope!r}")
    if not parameters:
        raise ValueError(
            f"gradient_clip_scope={scope!r} selected no parameters")
    return parameters

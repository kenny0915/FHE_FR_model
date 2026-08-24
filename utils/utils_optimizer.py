"""Optimizer parameter grouping helpers."""

from contextlib import contextmanager

import torch
import torch.nn as nn


_NORMALIZATION_TYPES = (
    nn.modules.batchnorm._BatchNorm,
    nn.GroupNorm,
    nn.LayerNorm,
    nn.LocalResponseNorm,
    nn.PReLU,
)


@contextmanager
def temporary_optimizer_lr_scale(optimizer, scale, scope=None):
    """Apply a temporary LR multiplier without changing scheduler state.

    When ``scope`` is provided, only parameter groups whose ``scope`` metadata
    matches are scaled.  This lets progressive-polynomial training slow the
    backbone while keeping the current activation coefficients and classifier
    at their scheduled learning rates.
    """
    scale = float(scale)
    if scale <= 0.0:
        raise ValueError("optimizer LR scale must be positive")
    selected = [
        group for group in optimizer.param_groups
        if scope is None or group.get("scope") == scope
    ]
    if scope is not None and not selected:
        raise ValueError(f"optimizer has no parameter groups for scope {scope!r}")
    original_lrs = [group["lr"] for group in selected]
    try:
        for group, original_lr in zip(selected, original_lrs):
            group["lr"] = original_lr * scale
        yield
    finally:
        for group, original_lr in zip(selected, original_lrs):
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
            if parameter.requires_grad and (
                    exclude_module or name in ("bias", "gain")):
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


@torch.no_grad()
def nonfinite_gradient_tensor_count(named_modules, device=None):
    """Return a device scalar counting gradient tensors with NaN or Inf.

    The count remains on-device, avoiding one CPU synchronization per model
    parameter on healthy training steps. ``named_modules`` may be iterated
    only once, so callers that later need details should pass it again.
    """
    count = None
    for _, module in named_modules:
        for parameter in module.parameters():
            gradient = parameter.grad
            if gradient is None:
                continue
            values = (
                gradient.detach().coalesce().values()
                if gradient.is_sparse else gradient.detach()
            )
            if count is None:
                count = torch.zeros((), device=values.device, dtype=torch.long)
            elif values.device != count.device:
                raise ValueError(
                    "Gradient finiteness check requires one gradient device")
            count.add_((~torch.isfinite(values)).any().to(dtype=torch.long))
    if count is None:
        count = torch.zeros((), device=device, dtype=torch.long)
    elif device is not None and count.device != torch.device(device):
        raise ValueError(
            "Requested gradient count device does not match gradient device")
    return count


@torch.no_grad()
def nonfinite_gradient_diagnostics(named_modules, max_diagnostics=4):
    """Count non-finite gradient tensors and describe the first few.

    ``named_modules`` is an iterable of ``(prefix, module)`` pairs. Sparse
    gradients are inspected through their coalesced values. The returned
    diagnostics are plain dictionaries so callers can log them without
    retaining gradient tensors or autograd graphs.
    """
    max_diagnostics = int(max_diagnostics)
    if max_diagnostics < 0:
        raise ValueError("max_diagnostics must be non-negative")

    nonfinite_tensors = 0
    diagnostics = []
    for module_name, module in named_modules:
        for parameter_name, parameter in module.named_parameters():
            gradient = parameter.grad
            if gradient is None:
                continue
            values = (
                gradient.detach().coalesce().values()
                if gradient.is_sparse else gradient.detach()
            )
            finite = torch.isfinite(values)
            if bool(finite.all()):
                continue
            nonfinite_tensors += 1
            if len(diagnostics) >= max_diagnostics:
                continue
            finite_values = values[finite]
            finite_absmax = (
                float(finite_values.abs().amax().item())
                if finite_values.numel() else float("nan")
            )
            diagnostics.append({
                "name": f"{module_name}.{parameter_name}",
                "shape": tuple(gradient.shape),
                "nonfinite_elements": int((~finite).sum().item()),
                "finite_absmax": finite_absmax,
            })
    return nonfinite_tensors, tuple(diagnostics)


def clip_grad_norm_stable(parameters, max_norm, error_if_nonfinite=False):
    """Clip an L2 gradient norm using FP64 reductions.

    ``torch.nn.utils.clip_grad_norm_`` accumulates FP32 squared gradients for
    FP32 parameters. A collection of individually finite, very large
    gradients can therefore produce an infinite total norm. Reducing each
    tensor and the final vector in FP64 distinguishes that overflow from a
    genuinely non-finite gradient and produces the correct clipping factor.
    """
    gradients = [
        parameter.grad for parameter in list(parameters)
        if parameter.grad is not None
    ]
    if not gradients:
        return torch.tensor(0.0)
    device = gradients[0].device
    if any(gradient.device != device for gradient in gradients):
        raise ValueError("Stable norm clipping requires one gradient device")
    norms = [
        torch.linalg.vector_norm(
            gradient.detach(), ord=2, dtype=torch.float64)
        for gradient in gradients
    ]
    total_norm = torch.linalg.vector_norm(
        torch.stack(norms), ord=2, dtype=torch.float64)
    if not torch.isfinite(total_norm):
        if error_if_nonfinite:
            raise RuntimeError(
                "The FP64 total gradient norm is non-finite and cannot be "
                "clipped")
        return total_norm
    clip_coefficient = float(max_norm) / (total_norm + 1e-12)
    clip_coefficient = clip_coefficient.clamp(max=1.0)
    for gradient in gradients:
        gradient.mul_(clip_coefficient.to(
            device=gradient.device, dtype=gradient.dtype))
    return total_norm

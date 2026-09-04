"""Gradient tools for accuracy/tail multi-objective recovery."""

from contextlib import contextmanager

import torch
import torch.nn as nn
from torch import distributed


@contextmanager
def temporary_batchnorm_eval(module):
    """Use checkpoint running statistics without freezing BN affine grads."""
    states = []
    for submodule in module.modules():
        if isinstance(submodule, nn.modules.batchnorm._BatchNorm):
            states.append((submodule, submodule.training))
            submodule.eval()
    try:
        yield
    finally:
        for submodule, was_training in states:
            submodule.train(was_training)


def combine_conflict_aware_gradients(
        parameters, clean_gradients, tail_gradients, *,
        learning_rate, tail_to_clean_ratio=1.0,
        max_step_update_ratio=1e-5, scale_floor=1.0, eps=1e-24):
    """Project conflicting tail gradients and cap every tensor's update.

    The tail component is projected off the clean component only when their
    dot product is negative, then limited relative to the clean gradient.
    Finally, the intended SGD update for each tensor is bounded relative to
    its parameter norm.  The function does not mutate parameters or grads.
    """
    parameters = tuple(parameters)
    clean_gradients = tuple(clean_gradients)
    tail_gradients = tuple(tail_gradients)
    if not (len(parameters) == len(clean_gradients) == len(tail_gradients)):
        raise ValueError("parameters and gradient lists must have equal length")
    learning_rate = float(learning_rate)
    tail_to_clean_ratio = float(tail_to_clean_ratio)
    max_step_update_ratio = float(max_step_update_ratio)
    scale_floor = float(scale_floor)
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    if tail_to_clean_ratio < 0.0:
        raise ValueError("tail_to_clean_ratio must be non-negative")
    if max_step_update_ratio <= 0.0:
        raise ValueError("max_step_update_ratio must be positive")
    if scale_floor <= 0.0:
        raise ValueError("scale_floor must be positive")

    combined = []
    conflict_flags = []
    tail_limit_flags = []
    update_limit_flags = []
    for parameter, clean_gradient, tail_gradient in zip(
            parameters, clean_gradients, tail_gradients):
        clean = (
            torch.zeros_like(parameter)
            if clean_gradient is None else clean_gradient.detach())
        tail = (
            torch.zeros_like(parameter)
            if tail_gradient is None else tail_gradient.detach())
        clean64 = clean.to(dtype=torch.float64)
        tail64 = tail.to(dtype=torch.float64)
        clean_norm = torch.linalg.vector_norm(clean64)
        tail_norm = torch.linalg.vector_norm(tail64)

        if clean_gradient is not None and tail_gradient is not None:
            dot = torch.sum(clean64 * tail64)
            conflict = dot < 0
            projection = torch.minimum(dot, dot.new_zeros(()))
            tail = tail - (
                projection / (clean_norm.square() + eps)).to(tail.dtype) * clean
            tail64 = tail.to(dtype=torch.float64)
            tail_norm = torch.linalg.vector_norm(tail64)
            conflict_flags.append(conflict)
        else:
            conflict_flags.append(clean_norm.new_zeros((), dtype=torch.bool))

        parameter_norm = torch.linalg.vector_norm(
            parameter.detach(), dtype=torch.float64)
        reference_norm = torch.maximum(
            parameter_norm,
            parameter_norm.new_tensor(scale_floor),
        )
        tail_limit = tail_to_clean_ratio * torch.maximum(
            clean_norm, reference_norm * 1e-6)
        tail_limit_flags.append(tail_norm > tail_limit)
        tail = tail * torch.clamp(
            tail_limit / (tail_norm + eps), max=1.0).to(tail.dtype)

        gradient = clean + tail
        gradient_norm = torch.linalg.vector_norm(
            gradient.detach(), dtype=torch.float64)
        gradient_limit = (
            max_step_update_ratio * reference_norm / learning_rate)
        update_limit_flags.append(gradient_norm > gradient_limit)
        gradient = gradient * torch.clamp(
            gradient_limit / (gradient_norm + eps), max=1.0).to(
                gradient.dtype)
        combined.append(gradient)

    counts = torch.stack((
        torch.stack(conflict_flags).sum(),
        torch.stack(tail_limit_flags).sum(),
        torch.stack(update_limit_flags).sum(),
    )).cpu().tolist()
    return tuple(combined), {
        "conflicts": int(counts[0]),
        "tail_limited": int(counts[1]),
        "update_limited": int(counts[2]),
        "tensor_count": len(parameters),
    }


@torch.no_grad()
def synchronize_and_assign_gradients(parameters, gradients):
    """Average dense gradients across ranks and assign ``parameter.grad``."""
    parameters = tuple(parameters)
    gradients = tuple(gradients)
    if len(parameters) != len(gradients):
        raise ValueError("parameters and gradients must have equal length")
    if not parameters:
        raise ValueError("at least one parameter is required")
    flat = torch.cat([gradient.reshape(-1) for gradient in gradients])
    if distributed.is_available() and distributed.is_initialized():
        distributed.all_reduce(flat, op=distributed.ReduceOp.SUM)
        flat.div_(distributed.get_world_size())
    offset = 0
    for parameter in parameters:
        count = parameter.numel()
        parameter.grad = flat[offset:offset + count].view_as(parameter).clone()
        offset += count


@torch.no_grad()
def project_to_relative_trust_region(
        parameters, anchors, ratio, scale_floor=1.0, eps=1e-24):
    """Project every tensor into an L2 ball around its initial value."""
    parameters = tuple(parameters)
    anchors = tuple(anchors)
    if len(parameters) != len(anchors):
        raise ValueError("parameters and anchors must have equal length")
    ratio = float(ratio)
    scale_floor = float(scale_floor)
    if ratio <= 0.0:
        raise ValueError("ratio must be positive")
    projected_flags = []
    relative_deltas = []
    for parameter, anchor in zip(parameters, anchors):
        delta = parameter.detach() - anchor
        anchor_norm = torch.linalg.vector_norm(anchor, dtype=torch.float64)
        reference_norm = torch.maximum(
            anchor_norm, anchor_norm.new_tensor(scale_floor))
        delta_norm = torch.linalg.vector_norm(delta, dtype=torch.float64)
        limit = ratio * reference_norm
        projected_flags.append(delta_norm > limit)
        projection_scale = torch.clamp(
            limit / (delta_norm + eps), max=1.0)
        parameter.copy_(
            anchor + delta * projection_scale.to(delta.dtype))
        relative_deltas.append(
            torch.minimum(delta_norm, limit) / reference_norm)
    projected, worst_relative_delta = torch.stack((
        torch.stack(projected_flags).sum(),
        torch.stack(relative_deltas).amax(),
    )).cpu().tolist()
    return {
        "projected": int(projected),
        "worst_relative_delta": float(worst_relative_delta),
    }

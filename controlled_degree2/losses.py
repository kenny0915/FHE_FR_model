"""Losses and schedules matched to the run10 polynomial distillation recipe."""

import math

import torch
from torch.nn import functional as F


def _piecewise(progress, values):
    if progress >= 1.0:
        return values[-1]
    return values[min(int(progress * len(values)), len(values) - 1)]


def gamma_at(step: int, warmup_steps: int, final: float = 10.0) -> float:
    if warmup_steps <= 0:
        return final
    return _piecewise(step / warmup_steps, (4.0, 6.0, 8.0, final))


def beta_at(step: int, warmup_steps: int, final: float) -> float:
    if warmup_steps <= 0:
        return final
    return final * _piecewise(step / warmup_steps, (0.01, 0.02, 0.1, 0.2, 1.0))


def hint_weight_at(step: int, total_steps: int, start: float, end: float) -> float:
    progress = min(step / max(total_steps, 1), 1.0)
    return end + 0.5 * (start - end) * (1.0 + math.cos(math.pi * progress))


def swap_alpha_at(
    step: int, index: int, layers: int, swap_steps: int, ramp_steps: int
) -> float:
    if swap_steps <= 0:
        return 1.0
    start = index * swap_steps / layers
    return float(min(max((step - start) / max(ramp_steps, 1), 0.0), 1.0))


def lr_factor(step: int, warmup_steps: int, total_steps: int, floor=0.01) -> float:
    if step < warmup_steps:
        return (step + 1) / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return floor + (1.0 - floor) * 0.5 * (
        1.0 + math.cos(math.pi * min(progress, 1.0))
    )


def embedding_loss(student, teacher, mask=None):
    loss = 1.0 - F.cosine_similarity(student.float(), teacher.float(), dim=1)
    if mask is None:
        return loss.mean()
    weights = mask.to(loss.dtype)
    return (loss * weights).sum() / weights.sum().clamp_min(1.0)


def hint_loss(student_store, teacher_store, names, mask=None):
    terms = []
    for name in names:
        student = student_store[name].float()
        teacher = teacher_store[name].detach().float()
        if mask is not None:
            student, teacher = student[mask], teacher[mask]
        terms.append(F.mse_loss(student, teacher) / (teacher.var() + 1e-6))
    return torch.stack(terms).mean()

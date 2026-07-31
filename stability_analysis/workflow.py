"""Core forward/backward simulation and layer-range collection."""

import copy
import dataclasses
import math
from collections import defaultdict

import torch
from torch import nn

from .activations import make_herpn_for


@dataclasses.dataclass
class AnalysisConfig:
    max_batches: int = 10
    interval: tuple = (-6.0, 6.0)
    backward: bool = True
    fail_fast: bool = False


def replace_activations(model, factory=None, input_scale=6.0):
    """Deep-copy ``model`` and replace every PReLU/ReLU/GELU activation.

    ``factory`` receives ``(qualified_name, original_module)`` and must return
    an ``nn.Module``.  If omitted, the repository's HerPN quadratic is used.
    """
    replaced = copy.deepcopy(model)
    replacements = {}

    def visit(parent, prefix=""):
        for child_name, child in list(parent.named_children()):
            name = f"{prefix}.{child_name}" if prefix else child_name
            if isinstance(child, (nn.PReLU, nn.ReLU, nn.GELU)):
                new_module = (
                    factory(name, child) if factory is not None
                    else make_herpn_for(child, input_scale=input_scale)
                )
                if not isinstance(new_module, nn.Module):
                    raise TypeError(f"activation factory returned {type(new_module)!r} for {name}")
                setattr(parent, child_name, new_module)
                replacements[name] = {
                    "original": child.__class__.__name__,
                    "replacement": new_module.__class__.__name__,
                    "target": getattr(new_module, "target", "unspecified"),
                    "interval": list(getattr(new_module, "interval", (-input_scale, input_scale))),
                    "degree": getattr(new_module, "degree", None),
                }
            else:
                visit(child, name)

    visit(replaced)
    return replaced, replacements


class _Stats:
    def __init__(self, interval):
        self.interval = interval
        self.data = defaultdict(lambda: {
            "count": 0, "nonfinite": 0, "outside": 0,
            "min": math.inf, "max": -math.inf, "absmax": 0.0,
            "sum": 0.0, "sum_sq": 0.0,
        })

    def add(self, name, value):
        value = value.detach().float()
        entry = self.data[name]
        entry["count"] += value.numel()
        finite = torch.isfinite(value)
        entry["nonfinite"] += (~finite).sum().item()
        clean = value[finite]
        if clean.numel() == 0:
            return
        low, high = self.interval
        entry["outside"] += ((clean < low) | (clean > high)).sum().item()
        entry["min"] = min(entry["min"], clean.min().item())
        entry["max"] = max(entry["max"], clean.max().item())
        entry["absmax"] = max(entry["absmax"], clean.abs().max().item())
        entry["sum"] += clean.double().sum().item()
        entry["sum_sq"] += clean.double().square().sum().item()

    def report(self):
        result = {}
        for name, entry in self.data.items():
            finite_count = entry["count"] - entry["nonfinite"]
            mean = entry["sum"] / finite_count if finite_count else None
            variance = max(entry["sum_sq"] / finite_count - mean * mean, 0.0) if finite_count else None
            result[name] = {
                "count": entry["count"],
                "nonfinite_count": entry["nonfinite"],
                "nonfinite_fraction": entry["nonfinite"] / entry["count"],
                "outside_interval_fraction": entry["outside"] / finite_count if finite_count else None,
                "min": entry["min"] if finite_count else None,
                "max": entry["max"] if finite_count else None,
                "absmax": entry["absmax"] if finite_count else None,
                "mean": mean,
                "std": math.sqrt(variance) if variance is not None else None,
            }
        return result


def analyze(baseline, polynomial, loader, replacements, device="cpu", config=None):
    """Compare baseline and substituted models without optimizer updates."""
    config = config or AnalysisConfig()
    baseline = baseline.to(device).eval()
    polynomial = polynomial.to(device).eval()
    stats = _Stats(config.interval)
    hooks = []
    modules = dict(polynomial.named_modules())
    for name in replacements:
        hooks.append(modules[name].register_forward_pre_hook(
            lambda _module, inputs, layer=name: stats.add(layer, inputs[0])))

    batches = 0
    output_mse = 0.0
    cosine_sum = 0.0
    backward_nonfinite = 0
    backward_tensors = 0
    output_nonfinite_batches = 0
    try:
        for batch in loader:
            if batches >= config.max_batches:
                break
            images = batch[0] if isinstance(batch, (tuple, list)) else batch
            images = images.to(device)
            with torch.no_grad():
                baseline_output = baseline(images).float()
            if config.backward:
                polynomial.zero_grad(set_to_none=True)
                polynomial_output = polynomial(images).float()
            else:
                with torch.no_grad():
                    polynomial_output = polynomial(images).float()
            finite = torch.isfinite(polynomial_output).all().item()
            output_nonfinite_batches += int(not finite)
            output_mse += torch.nan_to_num(
                (polynomial_output - baseline_output).square()).mean().item()
            cosine_sum += torch.nn.functional.cosine_similarity(
                torch.nan_to_num(polynomial_output), baseline_output, dim=1).mean().item()
            if config.backward:
                loss = polynomial_output.square().mean()
                loss.backward()
                for parameter in polynomial.parameters():
                    if parameter.grad is not None:
                        backward_tensors += 1
                        backward_nonfinite += int(not torch.isfinite(parameter.grad).all())
            batches += 1
            if config.fail_fast and not finite:
                break
    finally:
        for hook in hooks:
            hook.remove()
    layers = stats.report()
    unsafe = [name for name, values in layers.items()
              if values["nonfinite_count"] or (values["outside_interval_fraction"] or 0) > 0]
    return {
        "batches_analyzed": batches,
        "replacements": replacements,
        "activation_inputs": layers,
        "output": {
            "mse_vs_baseline": output_mse / batches if batches else None,
            "mean_cosine_similarity": cosine_sum / batches if batches else None,
            "nonfinite_batches": output_nonfinite_batches,
        },
        "backward_proxy": {
            "enabled": config.backward,
            "gradient_tensors": backward_tensors,
            "nonfinite_gradient_tensors": backward_nonfinite,
        },
        "summary": {
            "status": "warning" if unsafe or backward_nonfinite or output_nonfinite_batches else "pass",
            "layers_nonfinite_or_outside_interval": unsafe,
            "interpretation": "A pass applies only to sampled, no-update simulation; it is not a proof of training stability.",
        },
    }

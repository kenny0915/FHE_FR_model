"""Layer-by-layer activation replacement and interval sensitivity analysis."""

import math
from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F

from .activations import (
    evaluate_target_derivative_flat,
    evaluate_target_flat,
    make_herpn_for,
    make_uniform_quadratic_for,
)


SUPPORTED_ACTIVATIONS = (nn.PReLU, nn.ReLU, nn.GELU)


class TensorStats:
    """Streaming moments plus a bounded deterministic quantile sample."""

    def __init__(self, interval, max_samples=8192):
        self.interval = tuple(map(float, interval))
        self.max_samples = int(max_samples)
        self.count = 0
        self.nonfinite = 0
        self.outside = 0
        self.minimum = math.inf
        self.maximum = -math.inf
        self.sum = 0.0
        self.sum_sq = 0.0
        self._values = torch.empty(0)
        self._channels = torch.empty(0, dtype=torch.long)
        self._seen_finite = 0

    @staticmethod
    def _even_indices(count, selected, device="cpu"):
        if selected <= 0:
            return torch.empty(0, dtype=torch.long, device=device)
        if selected >= count:
            return torch.arange(count, device=device)
        return torch.linspace(
            0, count - 1, steps=selected, device=device).round().long()

    def add(self, value):
        value = value.detach().float()
        flat = value.reshape(-1)
        finite_mask = torch.isfinite(flat)
        self.count += flat.numel()
        self.nonfinite += int((~finite_mask).sum().item())
        clean = flat[finite_mask]
        if clean.numel() == 0:
            return

        low, high = self.interval
        self.outside += int(((clean < low) | (clean > high)).sum().item())
        self.minimum = min(self.minimum, float(clean.min().item()))
        self.maximum = max(self.maximum, float(clean.max().item()))
        self.sum += float(clean.double().sum().item())
        self.sum_sq += float(clean.double().square().sum().item())

        finite_indices = finite_mask.nonzero(as_tuple=False).flatten()
        previous_seen = self._seen_finite
        current_seen = finite_indices.numel()
        total_seen = previous_seen + current_seen
        self._seen_finite = total_seen
        if total_seen <= self.max_samples:
            current_target = current_seen
            previous_target = self._values.numel()
        else:
            current_target = min(
                current_seen,
                int(round(self.max_samples * current_seen / total_seen)))
            previous_target = min(
                self._values.numel(), self.max_samples - current_target)
            current_target = min(
                current_seen, self.max_samples - previous_target)

        previous_indices = self._even_indices(
            self._values.numel(), previous_target)
        current_positions = self._even_indices(
            current_seen, current_target, device=flat.device)
        selected_indices = finite_indices[current_positions]
        selected = flat[selected_indices].cpu()
        if value.ndim >= 2:
            values_per_channel = 1
            for dimension in value.shape[2:]:
                values_per_channel *= int(dimension)
            channels = (
                selected_indices // max(values_per_channel, 1)
            ) % value.shape[1]
        else:
            channels = torch.zeros_like(selected_indices)
        self._values = torch.cat((self._values[previous_indices], selected))
        self._channels = torch.cat((
            self._channels[previous_indices], channels.cpu().long()))

    def samples(self):
        if self._values.numel() == 0:
            return torch.empty(0), torch.empty(0, dtype=torch.long)
        return self._values, self._channels

    def report(self):
        finite_count = self.count - self.nonfinite
        mean = self.sum / finite_count if finite_count else None
        variance = (
            max(self.sum_sq / finite_count - mean * mean, 0.0)
            if finite_count else None
        )
        values, _ = self.samples()
        quantiles = {}
        if values.numel():
            levels = (0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 0.999)
            results = torch.quantile(values, torch.tensor(levels))
            quantiles = {
                "p{:g}".format(level * 100): float(result)
                for level, result in zip(levels, results)
            }
        return {
            "count": self.count,
            "sample_count": int(values.numel()),
            "nonfinite_count": self.nonfinite,
            "nonfinite_fraction": (
                self.nonfinite / self.count if self.count else None),
            "outside_interval_fraction": (
                self.outside / finite_count if finite_count else None),
            "min": self.minimum if finite_count else None,
            "max": self.maximum if finite_count else None,
            "absmax": (
                max(abs(self.minimum), abs(self.maximum))
                if finite_count else None),
            "mean": mean,
            "std": math.sqrt(variance) if variance is not None else None,
            "quantiles": quantiles,
        }


class ErrorStats:
    def __init__(self):
        self.count = 0
        self.sum_sq_error = 0.0
        self.sum_abs_error = 0.0
        self.sum_sq_target = 0.0
        self.max_abs_error = 0.0
        self.nonfinite = 0

    def add(self, target, actual):
        target = target.detach().float()
        actual = actual.detach().float()
        finite = torch.isfinite(target) & torch.isfinite(actual)
        self.count += target.numel()
        self.nonfinite += int((~finite).sum().item())
        if not finite.any():
            return
        target = target[finite]
        error = actual[finite] - target
        self.sum_sq_error += float(error.double().square().sum().item())
        self.sum_abs_error += float(error.double().abs().sum().item())
        self.sum_sq_target += float(target.double().square().sum().item())
        self.max_abs_error = max(
            self.max_abs_error, float(error.abs().max().item()))

    def report(self):
        finite_count = self.count - self.nonfinite
        return {
            "count": self.count,
            "nonfinite_count": self.nonfinite,
            "rmse": (
                math.sqrt(self.sum_sq_error / finite_count)
                if finite_count else None),
            "relative_rmse": (
                math.sqrt(self.sum_sq_error / max(self.sum_sq_target, 1e-30))
                if finite_count else None),
            "mean_absolute_error": (
                self.sum_abs_error / finite_count if finite_count else None),
            "max_absolute_error": self.max_abs_error if finite_count else None,
        }


class ObservedReplacement(nn.Module):
    """Return the polynomial while recording error against the teacher."""

    def __init__(self, teacher, polynomial):
        super().__init__()
        self.teacher = teacher
        self.polynomial = polynomial
        self.error = ErrorStats()

    def forward(self, inputs):
        teacher_output = self.teacher(inputs)
        polynomial_output = self.polynomial(inputs)
        self.error.add(teacher_output, polynomial_output)
        return polynomial_output


def activation_modules(model):
    return OrderedDict(
        (name, module) for name, module in model.named_modules()
        if isinstance(module, SUPPORTED_ACTIVATIONS)
    )


def _parent_and_child(model, qualified_name):
    pieces = qualified_name.split(".")
    parent = model
    for piece in pieces[:-1]:
        parent = parent._modules[piece]
    return parent, pieces[-1]


def _images(batch):
    return batch[0] if isinstance(batch, (tuple, list)) else batch


def _cache_batches(loader, max_batches):
    batches = []
    for batch in loader:
        if len(batches) >= max_batches:
            break
        batches.append(_images(batch).detach().cpu())
    return batches


def _run_baseline(model, batches, activation_items, device, interval,
                  max_samples):
    input_stats = {
        name: TensorStats(interval, max_samples) for name, _ in activation_items}
    output_stats = {
        name: TensorStats(interval, max_samples) for name, _ in activation_items}
    hooks = []
    for name, module in activation_items:
        hooks.append(module.register_forward_pre_hook(
            lambda _module, inputs, layer=name:
            input_stats[layer].add(inputs[0])))
        hooks.append(module.register_forward_hook(
            lambda _module, _inputs, output, layer=name:
            output_stats[layer].add(output)))
    outputs = []
    try:
        with torch.no_grad():
            for batch in batches:
                outputs.append(model(batch.to(device)).detach().float().cpu())
    finally:
        for hook in hooks:
            hook.remove()
    return input_stats, output_stats, outputs


def _embedding_behavior(baseline_outputs, polynomial_outputs):
    baseline = torch.cat(baseline_outputs)
    polynomial = torch.cat(polynomial_outputs)
    finite = torch.isfinite(polynomial)
    nonfinite_fraction = float((~finite).float().mean().item())
    clean_polynomial = torch.where(finite, polynomial, torch.zeros_like(polynomial))
    error = clean_polynomial - baseline
    baseline_energy = float(baseline.double().square().sum().item())
    cosine = F.cosine_similarity(clean_polynomial, baseline, dim=1)
    baseline_norm = baseline.norm(dim=1)
    polynomial_norm = clean_polynomial.norm(dim=1)

    pairwise_mae = None
    pairwise_max = None
    if baseline.shape[0] > 1:
        baseline_similarity = F.normalize(baseline, dim=1).mm(
            F.normalize(baseline, dim=1).t())
        polynomial_similarity = F.normalize(clean_polynomial, dim=1).mm(
            F.normalize(clean_polynomial, dim=1).t())
        mask = ~torch.eye(baseline.shape[0], dtype=torch.bool)
        pairwise_error = (
            polynomial_similarity - baseline_similarity)[mask].abs()
        pairwise_mae = float(pairwise_error.mean().item())
        pairwise_max = float(pairwise_error.max().item())

    return {
        "embedding_mse": float(error.square().mean().item()),
        "embedding_relative_rmse": math.sqrt(
            float(error.double().square().sum().item())
            / max(baseline_energy, 1e-30)),
        "embedding_cosine_mean": float(cosine.mean().item()),
        "embedding_cosine_min": float(cosine.min().item()),
        "embedding_norm_ratio_mean": float(
            (polynomial_norm / baseline_norm.clamp_min(1e-12)).mean().item()),
        "pairwise_cosine_mae": pairwise_mae,
        "pairwise_cosine_max_error": pairwise_max,
        "nonfinite_fraction": nonfinite_fraction,
    }


def _distribution_shift(reference, actual):
    reference = reference.report() if isinstance(reference, TensorStats) else reference
    actual = actual.report() if isinstance(actual, TensorStats) else actual
    reference_std = reference["std"] or 0.0
    actual_std = actual["std"] or 0.0
    return {
        "reference": reference,
        "replacement": actual,
        "mean_shift_in_reference_std": (
            abs(actual["mean"] - reference["mean"])
            / max(reference_std, 1e-12)),
        "std_ratio": actual_std / max(reference_std, 1e-12),
        "absmax_ratio": (
            actual["absmax"] / max(reference["absmax"], 1e-12)),
    }


def _derivative_metrics(teacher, polynomial, input_stats):
    values, channels = input_stats.samples()
    if values.numel() == 0:
        return {}
    target = evaluate_target_derivative_flat(teacher, values, channels)
    actual = polynomial.derivative_flat(values, channels)
    finite = torch.isfinite(actual)
    clean = actual[finite]
    error = actual[finite] - target[finite]
    return {
        "sample_count": int(values.numel()),
        "nonfinite_fraction": float((~finite).float().mean().item()),
        "mean": float(clean.mean().item()) if clean.numel() else None,
        "absmax": float(clean.abs().max().item()) if clean.numel() else None,
        "negative_fraction": (
            float((clean < 0).float().mean().item()) if clean.numel() else None),
        "rmse_vs_teacher": (
            float(error.square().mean().sqrt().item()) if clean.numel() else None),
    }


def _interval_metrics(teacher, input_stats, scales):
    values, channels = input_stats.samples()
    if values.numel() == 0:
        return []
    target = evaluate_target_flat(teacher, values, channels)
    target_energy = float(target.double().square().sum().item())
    rows = []
    for scale in scales:
        polynomial = make_uniform_quadratic_for(teacher, scale)
        actual = polynomial.evaluate_flat(values, channels)
        error = actual - target
        inside = values.abs() <= float(scale)
        outside = ~inside

        def subset_metrics(mask):
            if not mask.any():
                return {"count": 0, "rmse": None, "max_absolute_error": None}
            subset_error = error[mask]
            return {
                "count": int(mask.sum().item()),
                "rmse": float(subset_error.square().mean().sqrt().item()),
                "max_absolute_error": float(subset_error.abs().max().item()),
            }

        coefficients = polynomial.coefficients.detach().double()
        derivative = polynomial.derivative_flat(values, channels)
        rows.append({
            "interval": [-float(scale), float(scale)],
            "outside_fraction": float(outside.float().mean().item()),
            "observed_rmse": float(error.square().mean().sqrt().item()),
            "observed_relative_rmse": math.sqrt(
                float(error.double().square().sum().item())
                / max(target_energy, 1e-30)),
            "inside": subset_metrics(inside),
            "outside": subset_metrics(outside),
            "polynomial_output_absmax": float(actual.abs().max().item()),
            "coefficient_absmax": float(coefficients.abs().max().item()),
            "derivative_absmax": float(derivative.abs().max().item()),
            "negative_derivative_fraction": float(
                (derivative < 0).float().mean().item()),
        })
    return rows


def _replacement_metadata(module):
    coefficients = module.coefficients.detach().double()
    return {
        "class": module.__class__.__name__,
        "target": getattr(module, "target", "unspecified"),
        "interval": list(getattr(module, "interval", ())),
        "degree": getattr(module, "degree", None),
        "coefficient_absmax": float(coefficients.abs().max().item()),
        "constant_absmean": float(coefficients[0].abs().mean().item()),
        "coefficient_shape": list(coefficients.shape),
    }


def _summarize_intervals(layer_rows, scales):
    summaries = []
    for index, scale in enumerate(scales):
        rows = [layer["interval_sweep"][index] for layer in layer_rows]
        inside_rmse = [
            row["inside"]["rmse"] for row in rows
            if row["inside"]["rmse"] is not None]
        outside_rmse = [
            row["outside"]["rmse"] for row in rows
            if row["outside"]["rmse"] is not None]
        outside_max_error = [
            row["outside"]["max_absolute_error"] for row in rows
            if row["outside"]["max_absolute_error"] is not None]
        summaries.append({
            "interval": [-float(scale), float(scale)],
            "mean_outside_fraction": sum(
                row["outside_fraction"] for row in rows) / len(rows),
            "mean_observed_relative_rmse": sum(
                row["observed_relative_rmse"] for row in rows) / len(rows),
            "mean_inside_rmse": sum(inside_rmse) / len(inside_rmse),
            "mean_outside_rmse": (
                sum(outside_rmse) / len(outside_rmse)
                if outside_rmse else None),
            "worst_outside_absolute_error": (
                max(outside_max_error) if outside_max_error else None),
            "worst_polynomial_output_absmax": max(
                row["polynomial_output_absmax"] for row in rows),
            "worst_derivative_absmax": max(
                row["derivative_absmax"] for row in rows),
        })
    return summaries


def _run_all_replaced(model, batches, activation_items, baseline_outputs,
                      device, interval, max_samples, factory):
    restorations = []
    input_stats = {}
    hooks = []
    try:
        for name, teacher in activation_items:
            polynomial = factory(name, teacher).to(device).eval()
            parent, child_name = _parent_and_child(model, name)
            restorations.append((parent, child_name, teacher))
            parent._modules[child_name] = polynomial
            stats = TensorStats(interval, max_samples)
            input_stats[name] = stats
            hooks.append(polynomial.register_forward_pre_hook(
                lambda _module, inputs, layer=name:
                input_stats[layer].add(inputs[0])))

        outputs = []
        with torch.no_grad():
            for batch in batches:
                outputs.append(model(batch.to(device)).detach().float().cpu())
    finally:
        for parent, child_name, teacher in reversed(restorations):
            parent._modules[child_name] = teacher
        for hook in hooks:
            hook.remove()

    reports = {name: stats.report() for name, stats in input_stats.items()}
    unsafe = [
        name for name, stats in reports.items()
        if stats["nonfinite_count"] > 0]
    outside = [
        name for name, stats in reports.items()
        if (stats["outside_interval_fraction"] or 0.0) > 0]
    over_100 = [
        name for name, stats in reports.items()
        if stats["absmax"] is not None and stats["absmax"] > 100.0]
    return {
        "model_behavior": _embedding_behavior(baseline_outputs, outputs),
        "activation_inputs": reports,
        "layers_with_nonfinite_inputs": unsafe,
        "first_interval_violation": outside[0] if outside else None,
        "first_absmax_over_100": over_100[0] if over_100 else None,
        "first_nonfinite_input": unsafe[0] if unsafe else None,
    }


def analyze_layerwise(model, loader, device="cpu", max_batches=4,
                      interval=(-6.0, 6.0),
                      interval_scales=(0.5, 1, 2, 4, 6),
                      max_samples=8192, factory=None, progress=None):
    """Measure the isolated effect of replacing every activation once."""
    batches = _cache_batches(loader, max_batches)
    if not batches:
        raise ValueError("the data loader produced no batches")

    model = model.to(device).eval()
    activations = activation_modules(model)
    if not activations:
        raise ValueError("the model contains no supported activations")
    activation_items = list(activations.items())
    if progress is not None:
        progress("profiling baseline", 0, len(activation_items))
    input_stats, output_stats, baseline_outputs = _run_baseline(
        model, batches, activation_items, device, interval, max_samples)

    layer_rows = []
    for index, (name, teacher) in enumerate(activation_items):
        if progress is not None:
            progress(name, index + 1, len(activation_items))
        polynomial = (
            factory(name, teacher) if factory is not None
            else make_herpn_for(teacher, input_scale=max(abs(interval[0]), interval[1]))
        )
        polynomial = polynomial.to(device).eval()
        observed = ObservedReplacement(teacher, polynomial).to(device).eval()
        parent, child_name = _parent_and_child(model, name)
        next_name = (
            activation_items[index + 1][0]
            if index + 1 < len(activation_items) else None)
        next_stats = TensorStats(interval, max_samples)
        replacement_output_stats = TensorStats(interval, max_samples)
        hooks = [observed.register_forward_hook(
            lambda _module, _inputs, output:
            replacement_output_stats.add(output))]
        if next_name is not None:
            hooks.append(activations[next_name].register_forward_pre_hook(
                lambda _module, inputs: next_stats.add(inputs[0])))

        parent._modules[child_name] = observed
        polynomial_outputs = []
        try:
            with torch.no_grad():
                for batch in batches:
                    polynomial_outputs.append(
                        model(batch.to(device)).detach().float().cpu())
        finally:
            parent._modules[child_name] = teacher
            for hook in hooks:
                hook.remove()

        if next_name is None:
            baseline_embedding_stats = TensorStats(interval, max_samples)
            replacement_embedding_stats = TensorStats(interval, max_samples)
            for output in baseline_outputs:
                baseline_embedding_stats.add(output)
            for output in polynomial_outputs:
                replacement_embedding_stats.add(output)
            downstream_shift = _distribution_shift(
                baseline_embedding_stats, replacement_embedding_stats)
            downstream_name = "embedding"
        else:
            downstream_shift = _distribution_shift(
                input_stats[next_name], next_stats)
            downstream_name = next_name

        layer_rows.append({
            "index": index,
            "name": name,
            "stage": name.split(".")[0] if "." in name else "stem",
            "teacher": teacher.__class__.__name__,
            "replacement": _replacement_metadata(polynomial),
            "input_distribution": input_stats[name].report(),
            "teacher_output_distribution": output_stats[name].report(),
            "replacement_output_distribution": replacement_output_stats.report(),
            "local_approximation": observed.error.report(),
            "local_derivative": _derivative_metrics(
                teacher, polynomial, input_stats[name]),
            "model_behavior": _embedding_behavior(
                baseline_outputs, polynomial_outputs),
            "downstream_probe": downstream_name,
            "downstream_distribution_shift": downstream_shift,
            "interval_sweep": _interval_metrics(
                teacher, input_stats[name], interval_scales),
        })

    least_sensitive = sorted(
        layer_rows,
        key=lambda row: 1.0 - row["model_behavior"]["embedding_cosine_mean"])
    most_sensitive = list(reversed(least_sensitive))
    interval_summary = _summarize_intervals(layer_rows, interval_scales)
    best_interval = min(
        interval_summary, key=lambda row: row["mean_observed_relative_rmse"])
    if progress is not None:
        progress("all activations replaced", len(activation_items),
                 len(activation_items))
    replacement_factory = factory or (
        lambda _name, module: make_herpn_for(
            module, input_scale=max(abs(interval[0]), interval[1])))
    all_replaced = _run_all_replaced(
        model, batches, activation_items, baseline_outputs, device,
        interval, max_samples, replacement_factory)
    model.to("cpu")
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()

    return {
        "experiment": "isolated_single_activation_replacement",
        "batches_analyzed": len(batches),
        "samples_analyzed": sum(batch.shape[0] for batch in batches),
        "activation_count": len(layer_rows),
        "monitored_interval": list(map(float, interval)),
        "layer_results": layer_rows,
        "interval_summary": interval_summary,
        "all_replaced": all_replaced,
        "rankings": {
            "least_embedding_effect": [row["name"] for row in least_sensitive],
            "most_embedding_effect": [row["name"] for row in most_sensitive],
        },
        "data_driven_hints": {
            "lowest_mean_error_interval_in_this_sweep": best_interval["interval"],
            "first_replacement_candidates": [
                row["name"] for row in least_sensitive[:5]],
            "defer_or_use_longer_transition": [
                row["name"] for row in most_sensitive[:5]],
        },
        "limitations": [
            "Each row replaces one activation only; errors can compound nonlinearly.",
            "Interval fits are uniform-L2 quadratics, while the replacement rows use HerPN.",
            "Quantiles and interval errors use bounded deterministic activation samples.",
            "Embedding drift is not a substitute for verification accuracy.",
        ],
    }

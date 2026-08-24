#!/usr/bin/env python3
"""Record parameter and feature-map statistics for face backbones.

Feature-map statistics are aggregated online, so the amount of host memory does
not grow with the number of input images.  A small, configurable set of values
is retained for inspection and for approximate percentiles.
"""

import argparse
import csv
import json
import math
import os
import random
import sys
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backbones import get_model
from utils.utils_config import get_config


SUMMARY_FIELDS = (
    "kind", "layer", "tensor", "module_type", "shape", "calls", "numel",
    "finite", "nan", "posinf", "neginf", "min", "max", "mean", "std",
    "rms", "absmax", "abs_p99", "zero_fraction", "sample_values",
)


def _shape_text(shape, dynamic_batch=False):
    values = [str(value) for value in shape]
    if dynamic_batch and values:
        values[0] = "N"
    return "x".join(values) if values else "scalar"


class RunningTensorStats:
    """Online scalar statistics plus bounded representative values."""

    def __init__(self, kind, layer, tensor_name, module_type, sample_limit=128):
        self.kind = kind
        self.layer = layer
        self.tensor_name = tensor_name
        self.module_type = module_type
        self.sample_limit = max(0, int(sample_limit))
        self.shape = ""
        self.calls = 0
        self.numel = 0
        self.finite = 0
        self.nan = 0
        self.posinf = 0
        self.neginf = 0
        self.minimum = float("inf")
        self.maximum = float("-inf")
        self.total = 0.0
        self.total_square = 0.0
        self.zero = 0
        self.samples = torch.empty(0, dtype=torch.float32)
        self.sample_candidates_seen = 0
        self.sample_generator = random.Random(0)

    def update(self, tensor, dynamic_batch=False):
        if not torch.is_tensor(tensor):
            return
        with torch.no_grad():
            value = tensor.detach()
            if not self.shape:
                self.shape = _shape_text(value.shape, dynamic_batch)
            self.calls += 1
            self.numel += value.numel()
            if value.is_complex():
                value = value.abs()
            if value.dtype == torch.bool:
                value = value.to(torch.uint8)
            if not (value.is_floating_point() or value.dtype in (
                    torch.uint8, torch.int8, torch.int16, torch.int32,
                    torch.int64)):
                return

            flat = value.reshape(-1)
            if flat.is_floating_point():
                finite_mask = torch.isfinite(flat)
                self.nan += int(torch.isnan(flat).sum().item())
                self.posinf += int(torch.isposinf(flat).sum().item())
                self.neginf += int(torch.isneginf(flat).sum().item())
                flat = flat[finite_mask]
            if flat.numel() == 0:
                return

            # Keep reductions in FP32. Converting a full early-layer feature
            # map to FP64 can add hundreds of MB for ordinary IJB batch sizes.
            numeric = flat.float()
            self.finite += numeric.numel()
            batch_minimum = float(numeric.min().item())
            batch_maximum = float(numeric.max().item())
            batch_absmax = max(abs(batch_minimum), abs(batch_maximum))
            self.minimum = min(self.minimum, batch_minimum)
            self.maximum = max(self.maximum, batch_maximum)
            if batch_absmax:
                scaled = numeric / batch_absmax
                self.total += float(scaled.sum().item()) * batch_absmax
                self.total_square += (
                    float(torch.square(scaled).sum().item())
                    * batch_absmax * batch_absmax)
            self.zero += int((numeric == 0).sum().item())
            self._update_samples(numeric)

    def _update_samples(self, values):
        if self.sample_limit == 0:
            return
        values = values.reshape(-1)
        if values.numel() > self.sample_limit:
            indices = torch.linspace(
                0, values.numel() - 1, self.sample_limit,
                device=values.device).round().long()
            values = values[indices]
        values = values.cpu()
        available = self.sample_limit - self.samples.numel()
        if available > 0:
            initial = values[:available]
            self.samples = torch.cat((self.samples, initial))
            self.sample_candidates_seen += initial.numel()
            values = values[available:]
        for value in values:
            self.sample_candidates_seen += 1
            slot = self.sample_generator.randrange(self.sample_candidates_seen)
            if slot < self.sample_limit:
                self.samples[slot] = value

    def as_row(self):
        if self.finite:
            mean = self.total / self.finite
            variance = max(self.total_square / self.finite - mean * mean, 0.0)
            std = math.sqrt(variance)
            rms = math.sqrt(max(self.total_square / self.finite, 0.0))
            absmax = max(abs(self.minimum), abs(self.maximum))
            if self.samples.numel():
                abs_p99 = float(torch.quantile(
                    self.samples.abs(), 0.99).item())
            else:
                abs_p99 = float("nan")
            zero_fraction = self.zero / self.finite
            minimum = self.minimum
            maximum = self.maximum
        else:
            minimum = maximum = mean = std = rms = absmax = float("nan")
            abs_p99 = zero_fraction = float("nan")
        sample_values = [float(value) for value in self.samples.tolist()]
        return {
            "kind": self.kind,
            "layer": self.layer,
            "tensor": self.tensor_name,
            "module_type": self.module_type,
            "shape": self.shape,
            "calls": self.calls,
            "numel": self.numel,
            "finite": self.finite,
            "nan": self.nan,
            "posinf": self.posinf,
            "neginf": self.neginf,
            "min": minimum,
            "max": maximum,
            "mean": mean,
            "std": std,
            "rms": rms,
            "absmax": absmax,
            "abs_p99": abs_p99,
            "zero_fraction": zero_fraction,
            "sample_values": sample_values,
        }


def _named_output_tensors(value, prefix="output"):
    if torch.is_tensor(value):
        yield prefix, value
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            yield from _named_output_tensors(item, "{}.{}".format(prefix, index))
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from _named_output_tensors(item, "{}.{}".format(prefix, key))


class LayerStatisticsRecorder:
    """Attach hooks and aggregate output feature maps from model modules."""

    def __init__(self, model, sample_limit=128, module_scope="leaf",
                 capture="output"):
        self.model = model
        self.sample_limit = int(sample_limit)
        self.module_scope = module_scope
        self.capture = capture
        self.stats = OrderedDict()
        self.handles = []
        for name, module in model.named_modules():
            if not name:
                continue
            if module_scope == "leaf" and any(module.children()):
                continue
            self.handles.append(module.register_forward_hook(
                self._make_hook(name, module.__class__.__name__)))

    def _update(self, layer, tensor_name, module_type, tensor):
        key = (layer, tensor_name)
        if key not in self.stats:
            self.stats[key] = RunningTensorStats(
                "feature_map", layer, tensor_name, module_type,
                self.sample_limit)
        self.stats[key].update(tensor, dynamic_batch=True)

    def _make_hook(self, name, module_type):
        def hook(module, inputs, output):
            if self.capture in ("input", "both"):
                for tensor_name, tensor in _named_output_tensors(inputs, "input"):
                    self._update(name, tensor_name, module_type, tensor)
            if self.capture in ("output", "both"):
                for tensor_name, tensor in _named_output_tensors(output, "output"):
                    self._update(name, tensor_name, module_type, tensor)
        return hook

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def rows(self):
        return [item.as_row() for item in self.stats.values()]


def parameter_rows(model, sample_limit=128):
    rows = []
    for layer_name, module in model.named_modules():
        display_name = layer_name or "<root>"
        for parameter_name, parameter in module.named_parameters(recurse=False):
            item = RunningTensorStats(
                "parameter", display_name, parameter_name,
                module.__class__.__name__, sample_limit)
            item.update(parameter)
            rows.append(item.as_row())
    return rows


def extract_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint must be a state dictionary")
    for key in ("state_dict_backbone", "state_dict", "model"):
        nested = checkpoint.get(key)
        if isinstance(nested, dict):
            checkpoint = nested
            break
    state = OrderedDict()
    for key, value in checkpoint.items():
        if torch.is_tensor(value):
            state[key[7:] if key.startswith("module.") else key] = value
    if not state:
        raise ValueError("Checkpoint contains no tensor state")
    return state


def model_kwargs_from_config(cfg):
    """Recreate architecture-affecting arguments used by train_v2.py."""
    kwargs = {
        "dropout": 0.0,
        "fp16": False,
        "num_features": int(cfg.embedding_size),
    }
    if hasattr(cfg, "arch_config"):
        kwargs["arch_config"] = str(cfg.arch_config)
    network = cfg.network
    if network.startswith("r") and network.endswith(("_no_relu", "_prelu_herpn")):
        default_progress = 0.0 if (
            getattr(cfg, "herpn_conversion_groups", ())
            or getattr(cfg, "herpn_stage_epochs", ())) else 5.0
        kwargs.update(
            herpn_range_limit=float(getattr(cfg, "herpn_range_limit", 6.0)),
            herpn_bn_eps=float(getattr(cfg, "herpn_bn_eps", 1e-4)),
            herpn_progress=float(getattr(
                cfg, "herpn_initial_progress", default_progress)),
        )
        if network.endswith("_prelu_herpn"):
            kwargs["prelu_herpn_distill_eps"] = float(getattr(
                cfg, "prelu_herpn_distill_eps", 1e-4))
            if hasattr(cfg, "prelu_herpn_layerwise_scale"):
                kwargs["prelu_herpn_layerwise_scale"] = bool(
                    cfg.prelu_herpn_layerwise_scale)
                kwargs["prelu_herpn_initial_scale"] = float(getattr(
                    cfg, "prelu_herpn_initial_scale", 1.0))
    if network.startswith("r") and network.endswith("_herpn_residual_scale"):
        kwargs.update(
            herpn_range_limit=float(getattr(cfg, "herpn_range_limit", 6.0)),
            herpn_bn_eps=float(getattr(cfg, "herpn_bn_eps", 1e-4)),
            residual_scale_init=float(getattr(
                cfg, "residual_scale_init", 1.0 / math.sqrt(24.0))),
            residual_scale_trainable=bool(getattr(
                cfg, "residual_scale_trainable", True)),
        )
    if network.startswith("r") and network.endswith("_quadratic"):
        kwargs.update(
            quadratic_input_scale=float(getattr(
                cfg, "quadratic_input_scale", 6.0)),
            quadratic_range_limit=float(getattr(
                cfg, "quadratic_range_limit", 6.0)),
            quadratic_abs_init=float(getattr(
                cfg, "quadratic_abs_init", 1.0 / math.sqrt(2.0 * math.pi))),
            quadratic_progress=float(getattr(
                cfg, "herpn_initial_progress", 0.0)),
        )
    if network.startswith("r") and network.endswith("_layerwise_poly"):
        kwargs.update(
            layerwise_poly_degree=int(getattr(
                cfg, "layerwise_poly_degree", 2)),
            layerwise_poly_initial_scale=float(getattr(
                cfg, "layerwise_poly_initial_scale", 1.0)),
            layerwise_poly_distill_eps=float(getattr(
                cfg, "layerwise_poly_distill_eps", 1e-4)),
            layerwise_poly_progress=float(getattr(
                cfg, "herpn_initial_progress", 0.0)),
        )
    if network.startswith("r") and "_precise_relu" in network:
        alpha7 = network.endswith("_precise_relu_alpha7")
        kwargs.update(
            precise_relu_input_scale=float(getattr(
                cfg, "precise_relu_input_scale", 8.0)),
            precise_relu_target_alphas=tuple(getattr(
                cfg, "precise_relu_target_alphas", (7,) if alpha7 else ())),
            precise_relu_lower_degrees=tuple(getattr(
                cfg, "precise_relu_lower_degrees",
                () if alpha7 else (16, 8, 4))),
            precise_relu_progress=float(getattr(
                cfg, "precise_relu_initial_progress", 0.0)),
            precise_relu_backward_mode=str(getattr(
                cfg, "precise_relu_backward_mode", "exact")),
        )
    if network == "patch_cnn":
        kwargs.update(
            input_size=int(getattr(cfg, "input_size", 112)),
            patch_size=int(getattr(cfg, "patch_size", 28)),
        )
    if network.startswith("poolformer_no_ln_x2_act"):
        group_epochs = tuple(getattr(cfg, "simple_gate_group_epochs", ()))
        kwargs.update(
            gate_range_limit=float(getattr(
                cfg, "simple_gate_range_limit", 6.0)),
            gate_stats_sample_size=int(getattr(
                cfg, "simple_gate_stats_sample_size", 16384)),
            gate_compute_fp32=bool(getattr(
                cfg, "simple_gate_compute_fp32", True)),
            gate_fail_on_nonfinite=bool(getattr(
                cfg, "simple_gate_fail_on_nonfinite", True)),
            gate_initial_blend=float(getattr(
                cfg, "simple_gate_initial_blend",
                0.0 if group_epochs else 1.0)),
            gate_grouping=str(getattr(
                cfg, "simple_gate_grouping", "stage_chunks")),
        )
    if network.startswith("poolformer_fully_gated_prepbn"):
        kwargs.update(
            repbn_bn_eps=float(getattr(cfg, "repbn_bn_eps", 1e-5)),
            repbn_bn_momentum=float(getattr(
                cfg, "repbn_bn_momentum", 0.1)),
            repbn_eta_init=float(getattr(cfg, "repbn_eta_init", 0.0)),
        )
    if network.startswith("poolformer_fully_gated_affine"):
        kwargs["affine_blocks_per_group"] = int(getattr(
            cfg, "affine_blocks_per_group", 1))
    if network.startswith((
            "poolformer_fully_gated_frozen_std",
            "poolformer_fully_gated_spatial_frozen_std")):
        kwargs.update(
            frozen_std_momentum=float(getattr(
                cfg, "frozen_std_momentum", 0.9)),
            frozen_std_initial=float(getattr(
                cfg, "frozen_std_initial", 1.0)),
        )
    if network.startswith(("poolformer_nf", "iresnet_nf")):
        kwargs.update(
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
    return kwargs


def load_model(checkpoint_path, config_path, network, embedding_size,
               model_kwargs_json, device):
    if config_path:
        cfg = get_config(config_path)
        config_network = cfg.network
        network = network or config_network
        if network == config_network:
            kwargs = model_kwargs_from_config(cfg)
        else:
            kwargs = {
                "dropout": 0.0,
                "fp16": False,
                "num_features": int(cfg.embedding_size),
            }
    else:
        network = network or "r50"
        kwargs = {
            "dropout": 0.0,
            "fp16": False,
            "num_features": int(embedding_size),
        }
    if model_kwargs_json:
        overrides = json.loads(model_kwargs_json)
        if not isinstance(overrides, dict):
            raise ValueError("--model-kwargs must decode to a JSON object")
        kwargs.update(overrides)
    kwargs["fp16"] = False
    model = get_model(network, **kwargs)
    raw_checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = extract_state_dict(raw_checkpoint)
    model.load_state_dict(state, strict=True)
    return model.float().to(device).eval(), network


def estimate_similarity_transform(source, destination):
    """Return a 2-D similarity transform mapping source to destination.

    This is the least-squares Umeyama solution used for face alignment.  It is
    kept local so the statistics collector does not require scikit-image.
    """
    source = np.asarray(source, dtype=np.float64)
    destination = np.asarray(destination, dtype=np.float64)
    if source.shape != destination.shape or source.ndim != 2:
        raise ValueError("source and destination must have the same 2-D shape")
    if source.shape[0] < 2 or source.shape[1] != 2:
        raise ValueError("at least two 2-D point pairs are required")
    if not (np.isfinite(source).all() and np.isfinite(destination).all()):
        raise ValueError("alignment points must be finite")

    source_mean = source.mean(axis=0)
    destination_mean = destination.mean(axis=0)
    source_centered = source - source_mean
    destination_centered = destination - destination_mean
    source_variance = np.mean(np.sum(source_centered ** 2, axis=1))
    if source_variance <= np.finfo(np.float64).eps:
        raise ValueError("source alignment points are degenerate")

    covariance = destination_centered.T @ source_centered / source.shape[0]
    left, singular_values, right_transpose = np.linalg.svd(covariance)
    signs = np.ones(source.shape[1], dtype=np.float64)
    if np.linalg.det(left) * np.linalg.det(right_transpose) < 0:
        signs[-1] = -1.0
    rotation = (left * signs) @ right_transpose
    scale = float(np.dot(singular_values, signs) / source_variance)
    translation = destination_mean - scale * rotation @ source_mean
    matrix = np.column_stack((scale * rotation, translation))
    if not np.isfinite(matrix).all():
        raise ValueError("could not estimate a finite alignment transform")
    return matrix.astype(np.float32)


def align_ijb_image(image, landmark, use_flip=False):
    source = np.array([
        [30.2946, 51.6963],
        [65.5318, 51.5014],
        [48.0252, 71.7366],
        [33.5493, 92.3655],
        [62.7299, 92.2041],
    ], dtype=np.float32)
    source[:, 0] += 8.0
    matrix = estimate_similarity_transform(landmark, source)
    aligned = cv2.warpAffine(
        image, matrix, (112, 112), borderValue=0.0)
    aligned = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
    images = [aligned]
    if use_flip:
        images.append(np.fliplr(aligned))
    batch = np.stack(images).transpose(0, 3, 1, 2).astype(np.float32)
    return batch / 127.5 - 1.0


def iter_ijb_batches(ijb_root, target, batch_size, num_images,
                     use_flip=False):
    """Yield tensors and source-image counts from IJBB/IJBC loose crops."""
    target_lower = target.lower()
    metadata_path = os.path.join(
        ijb_root, "meta", "{}_name_5pts_score.txt".format(target_lower))
    crop_dir = os.path.join(ijb_root, "loose_crop")
    if not os.path.isfile(metadata_path):
        raise FileNotFoundError("IJB metadata not found: {}".format(metadata_path))
    pending = []
    source_count = 0
    with open(metadata_path, encoding="utf-8") as metadata:
        for line_number, line in enumerate(metadata, 1):
            if num_images > 0 and source_count >= num_images:
                break
            fields = line.strip().split()
            if len(fields) < 11:
                raise ValueError(
                    "Invalid IJB metadata at line {}: expected image path and "
                    "10 landmark coordinates".format(line_number))
            image_path = os.path.join(crop_dir, fields[0])
            image = cv2.imread(image_path)
            if image is None:
                print("warning: could not read {}; skipping".format(image_path),
                      file=sys.stderr)
                continue
            landmark = np.asarray(fields[1:11], dtype=np.float32).reshape(5, 2)
            pending.extend(align_ijb_image(image, landmark, use_flip))
            source_count += 1
            while len(pending) >= batch_size:
                values = np.stack(pending[:batch_size])
                del pending[:batch_size]
                yield torch.from_numpy(values), source_count
    if pending:
        yield torch.from_numpy(np.stack(pending)), source_count
    if num_images > 0 and source_count < num_images:
        print("warning: requested {} images but only {} readable images were "
              "found".format(num_images, source_count), file=sys.stderr)


def iter_synthetic_batches(batch_size, num_images):
    remaining = num_images
    processed = 0
    generator = torch.Generator().manual_seed(1234)
    while remaining > 0:
        current = min(batch_size, remaining)
        batch = torch.empty(current, 3, 112, 112)
        batch.uniform_(-1.0, 1.0, generator=generator)
        processed += current
        remaining -= current
        yield batch, processed


def _format_number(value):
    if isinstance(value, float):
        return "{:.6g}".format(value)
    return str(value)


def print_rows(rows):
    headers = ("kind", "layer.tensor", "type", "shape", "numel", "min",
               "max", "mean", "std", "abs_p99", "sample")
    rendered = []
    for row in rows:
        samples = row["sample_values"][:4]
        rendered.append((
            row["kind"], "{}.{}".format(row["layer"], row["tensor"]),
            row["module_type"], row["shape"], str(row["numel"]),
            _format_number(row["min"]), _format_number(row["max"]),
            _format_number(row["mean"]), _format_number(row["std"]),
            _format_number(row["abs_p99"]),
            "[{}]".format(", ".join(_format_number(x) for x in samples)),
        ))
    widths = [len(header) for header in headers]
    for row in rendered:
        widths = [max(width, len(value)) for width, value in zip(widths, row)]
    line = " | ".join(header.ljust(width) for header, width in zip(headers, widths))
    print(line)
    print("-+-".join("-" * width for width in widths))
    for row in rendered:
        print(" | ".join(value.ljust(width) for value, width in zip(row, widths)))


def write_results(rows, output_prefix, metadata):
    output_path = Path(output_prefix)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output_path.with_suffix(".csv")
    json_path = output_path.with_suffix(".json")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in rows:
            csv_row = dict(row)
            csv_row["sample_values"] = json.dumps(row["sample_values"])
            writer.writerow(csv_row)
    def json_safe(value):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if isinstance(value, dict):
            return {key: json_safe(item) for key, item in value.items()}
        if isinstance(value, list):
            return [json_safe(item) for item in value]
        return value

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(json_safe({"metadata": metadata, "statistics": rows}),
                  handle, indent=2, allow_nan=False)
        handle.write("\n")
    return csv_path, json_path


def build_parser():
    parser = argparse.ArgumentParser(
        description="Record per-layer weight and feature-map statistics.")
    parser.add_argument("--checkpoint", required=True,
                        help="Backbone model.pt or training checkpoint.")
    parser.add_argument(
        "--config", default="",
        help="Training config used for this checkpoint, e.g. configs/ms1mv3_r50.py.")
    parser.add_argument(
        "--network", default="",
        help="Backbone name. Overrides config; defaults to r50 without a config.")
    parser.add_argument("--embedding-size", type=int, default=512)
    parser.add_argument(
        "--model-kwargs", default="",
        help='JSON model-construction overrides, e.g. \'{"arch_config":"nl13"}\'.')
    parser.add_argument("--ijb-root", default="ijb/IJBC",
                        help="Dataset root containing meta/ and loose_crop/.")
    parser.add_argument("--target", choices=("IJBB", "IJBC"), default="IJBC")
    parser.add_argument("--num-images", type=int, default=1000,
                        help="Number of source pictures; 0 means all pictures.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--use-flip", action="store_true",
                        help="Also run the horizontal flip of every source picture.")
    parser.add_argument("--synthetic", action="store_true",
                        help="Use deterministic random inputs for a lightweight smoke test.")
    parser.add_argument("--device", default=(
        "cuda:0" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--sample-values", type=int, default=128,
                        help="Maximum representative values retained per tensor; 0 disables.")
    parser.add_argument("--module-scope", choices=("leaf", "all"), default="leaf",
                        help="Record leaf layers only, or every named module.")
    parser.add_argument("--capture", choices=("input", "output", "both"),
                        default="output", help="Feature maps to record at each layer.")
    parser.add_argument(
        "--output", default="work_dirs/layer_statistics",
        help="Output prefix; .csv and .json are written.")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.num_images < 0:
        raise ValueError("--num-images must be non-negative")
    if args.sample_values < 0:
        raise ValueError("--sample-values must be non-negative")
    if args.synthetic and args.num_images == 0:
        raise ValueError("--synthetic requires --num-images greater than zero")

    device = torch.device(args.device)
    model, network = load_model(
        args.checkpoint, args.config, args.network, args.embedding_size,
        args.model_kwargs, device)
    print("Loaded {} from {} on {}".format(network, args.checkpoint, device))
    weights = parameter_rows(model, args.sample_values)
    recorder = LayerStatisticsRecorder(
        model, args.sample_values, args.module_scope, args.capture)
    batches = (
        iter_synthetic_batches(args.batch_size, args.num_images)
        if args.synthetic else
        iter_ijb_batches(args.ijb_root, args.target, args.batch_size,
                         args.num_images, args.use_flip)
    )
    processed_images = 0
    processed_inputs = 0
    try:
        with torch.inference_mode():
            for batch_index, (batch, source_count) in enumerate(batches, 1):
                model(batch.to(device, non_blocking=True))
                processed_images = source_count
                processed_inputs += batch.shape[0]
                print("Processed batch {}: {} source images, {} model inputs".format(
                    batch_index, processed_images, processed_inputs), flush=True)
    finally:
        recorder.close()
    if processed_images == 0:
        raise RuntimeError("No input images were processed")
    rows = weights + recorder.rows()
    metadata = {
        "checkpoint": str(args.checkpoint),
        "config": str(args.config),
        "network": network,
        "dataset": "synthetic" if args.synthetic else str(args.ijb_root),
        "target": args.target,
        "source_images": processed_images,
        "model_inputs": processed_inputs,
        "use_flip": bool(args.use_flip),
        "module_scope": args.module_scope,
        "capture": args.capture,
        "sample_values_per_tensor": args.sample_values,
    }
    csv_path, json_path = write_results(rows, args.output, metadata)
    print_rows(rows)
    print("Wrote {}".format(csv_path))
    print("Wrote {}".format(json_path))
    return rows, metadata


if __name__ == "__main__":
    main()

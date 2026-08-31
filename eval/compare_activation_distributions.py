#!/usr/bin/env python3
"""Compare activation tails for two face-backbone checkpoints.

This utility is intentionally validation-bin based so the same aligned public
images are presented to both graphs. It records wrapper inputs/outputs and the
final embedding, including non-finite counts and tail percentiles. Percentiles
come from a bounded reservoir; extrema and moments cover every processed
value.
"""

import argparse
import csv
import json
import math
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backbones import get_model
from eval.layer_statistics import RunningTensorStats, extract_state_dict


FIELDS = (
    "layer", "tensor", "baseline_std", "candidate_std", "std_ratio",
    "baseline_abs_p99", "candidate_abs_p99", "abs_p99_ratio",
    "baseline_abs_p999", "candidate_abs_p999", "abs_p999_ratio",
    "baseline_absmax", "candidate_absmax", "absmax_ratio",
    "baseline_nonfinite", "candidate_nonfinite",
)


def _is_progressive_wrapper(module):
    return (
        module.__class__.__name__ == "ProgressiveHerPNActivation"
        or getattr(module, "is_progressive_polynomial_activation", False)
    )


class ActivationDistributionRecorder:
    def __init__(self, model, sample_limit=65536):
        wrappers = [
            (name, module) for name, module in model.named_modules()
            if _is_progressive_wrapper(module)
        ]
        modules = wrappers or [
            (name, module) for name, module in model.named_modules()
            if isinstance(module, nn.PReLU)
        ]
        self.stats = {}
        self.handles = []
        self.sample_limit = int(sample_limit)
        for name, module in modules:
            self.handles.append(module.register_forward_hook(
                self._make_hook(name, module.__class__.__name__)))
        self.stats[("embedding", "output")] = RunningTensorStats(
            "feature_map", "embedding", "output", "ModelOutput",
            sample_limit=self.sample_limit)

    def _stat(self, layer, tensor, module_type):
        key = (layer, tensor)
        if key not in self.stats:
            self.stats[key] = RunningTensorStats(
                "feature_map", layer, tensor, module_type,
                sample_limit=self.sample_limit)
        return self.stats[key]

    def _make_hook(self, name, module_type):
        def hook(module, inputs, output):
            self._stat(name, "input", module_type).update(
                inputs[0], dynamic_batch=True)
            self._stat(name, "output", module_type).update(
                output, dynamic_batch=True)
        return hook

    def update_embedding(self, embedding):
        self.stats[("embedding", "output")].update(
            embedding, dynamic_batch=True)

    def rows(self):
        rows = {}
        for key, stats in self.stats.items():
            row = stats.as_row()
            samples = stats.samples.abs()
            row["abs_p999"] = (
                float(torch.quantile(samples, 0.999).item())
                if samples.numel() else float("nan")
            )
            row["nonfinite"] = row["nan"] + row["posinf"] + row["neginf"]
            rows[key] = row
        return rows

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []


def load_model(network, checkpoint, device, model_kwargs=None):
    kwargs = dict(model_kwargs or {})
    kwargs.update(dropout=0.0, fp16=False, num_features=512)
    model = get_model(network, **kwargs)
    raw = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(extract_state_dict(raw), strict=True)
    return model.float().to(device).eval()


def iter_bin_batches(path, batch_size, max_images):
    with open(path, "rb") as handle:
        encoded, _ = pickle.load(handle, encoding="bytes")
    if max_images > 0:
        encoded = encoded[:max_images]
    pending = []
    for index, value in enumerate(encoded, 1):
        image = cv2.imdecode(
            np.frombuffer(value, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not decode image {index - 1} from {path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pending.append(image.transpose(2, 0, 1))
        if len(pending) == batch_size:
            batch = np.stack(pending).astype(np.float32) / 127.5 - 1.0
            yield torch.from_numpy(batch)
            pending = []
    if pending:
        batch = np.stack(pending).astype(np.float32) / 127.5 - 1.0
        yield torch.from_numpy(batch)


@torch.no_grad()
def profile(model, batches, device, sample_limit):
    recorder = ActivationDistributionRecorder(
        model, sample_limit=sample_limit)
    try:
        for batch_index, batch in enumerate(batches):
            embedding = model(batch.to(device=device, non_blocking=True))
            recorder.update_embedding(embedding)
            print(f"profile batch {batch_index}", flush=True)
        return recorder.rows()
    finally:
        recorder.close()


def _ratio(candidate, baseline):
    if baseline == 0.0:
        return float("inf") if candidate != 0.0 else 1.0
    return candidate / baseline


def compare_rows(baseline, candidate):
    rows = []
    for key in sorted(set(baseline).intersection(candidate)):
        base = baseline[key]
        new = candidate[key]
        rows.append({
            "layer": key[0],
            "tensor": key[1],
            "baseline_std": base["std"],
            "candidate_std": new["std"],
            "std_ratio": _ratio(new["std"], base["std"]),
            "baseline_abs_p99": base["abs_p99"],
            "candidate_abs_p99": new["abs_p99"],
            "abs_p99_ratio": _ratio(new["abs_p99"], base["abs_p99"]),
            "baseline_abs_p999": base["abs_p999"],
            "candidate_abs_p999": new["abs_p999"],
            "abs_p999_ratio": _ratio(new["abs_p999"], base["abs_p999"]),
            "baseline_absmax": base["absmax"],
            "candidate_absmax": new["absmax"],
            "absmax_ratio": _ratio(new["absmax"], base["absmax"]),
            "baseline_nonfinite": base["nonfinite"],
            "candidate_nonfinite": new["nonfinite"],
        })
    return rows


def _json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def main():
    parser = argparse.ArgumentParser(
        description="Compare activation distributions on an aligned .bin set")
    parser.add_argument("--baseline-checkpoint", required=True)
    parser.add_argument("--candidate-checkpoint", required=True)
    parser.add_argument("--baseline-network", default="r50")
    parser.add_argument("--candidate-network", default="r50_no_relu")
    parser.add_argument("--baseline-model-kwargs", default="{}")
    parser.add_argument("--candidate-model-kwargs", default="{}")
    parser.add_argument("--data-bin", default="ms1m-retinaface-t1/cfp_fp.bin")
    parser.add_argument("--max-images", type=int, default=1400)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--sample-limit", type=int, default=65536)
    parser.add_argument("--output-prefix", default="work_dirs/activation_compare")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    model_kwargs = {
        "baseline": json.loads(args.baseline_model_kwargs),
        "candidate": json.loads(args.candidate_model_kwargs),
    }
    if not all(isinstance(value, dict) for value in model_kwargs.values()):
        raise ValueError("model kwargs must decode to JSON objects")
    profiles = {}
    for label, network, checkpoint in (
            ("baseline", args.baseline_network, args.baseline_checkpoint),
            ("candidate", args.candidate_network, args.candidate_checkpoint)):
        print(f"loading {label}: {checkpoint}", flush=True)
        model = load_model(
            network, checkpoint, device, model_kwargs=model_kwargs[label])
        batches = iter_bin_batches(
            args.data_bin, args.batch_size, args.max_images)
        profiles[label] = profile(
            model, batches, device, args.sample_limit)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    rows = compare_rows(profiles["baseline"], profiles["candidate"])
    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    with prefix.with_suffix(".csv").open(
            "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    with prefix.with_suffix(".json").open("w", encoding="utf-8") as handle:
        json.dump(_json_safe({
            "metadata": vars(args),
            "comparison": rows,
        }), handle, indent=2, allow_nan=False)
        handle.write("\n")

    print("layer,tensor,std_ratio,abs_p999_ratio,absmax_ratio,nonfinite")
    for row in rows:
        print(
            f'{row["layer"]},{row["tensor"]},{row["std_ratio"]:.6g},'
            f'{row["abs_p999_ratio"]:.6g},{row["absmax_ratio"]:.6g},'
            f'{row["candidate_nonfinite"]}')


if __name__ == "__main__":
    main()

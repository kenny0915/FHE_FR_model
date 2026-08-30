#!/usr/bin/env python3
"""Locate the first unstable PILLAR activation for verification-bin rows."""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backbones import get_model
from backbones.iresnet_pillar import PILLARPolynomialReLU
from eval import verification
from eval.layer_statistics import extract_state_dict
from utils.utils_config import get_config


def tensor_summary(value):
    value = value.detach()
    finite = torch.isfinite(value)
    finite_values = value[finite].float()
    return {
        "numel": value.numel(),
        "finite": int(finite.sum().item()),
        "nan": int(torch.isnan(value).sum().item()),
        "posinf": int(torch.isposinf(value).sum().item()),
        "neginf": int(torch.isneginf(value).sum().item()),
        "finite_absmax": (
            float(finite_values.abs().amax().item())
            if finite_values.numel() else None),
    }


def first_nonfinite_activation(trace):
    for index, item in enumerate(trace):
        if item["output"]["finite"] != item["output"]["numel"]:
            return index
    return None


def compact_activation_trace(trace):
    """Keep the range cascade readable while retaining every PILLAR site."""
    return [
        {
            "name": item["name"],
            "input_absmax": item["input"]["finite_absmax"],
            "output_absmax": item["output"]["finite_absmax"],
            "output_nonfinite": (
                item["output"]["numel"] - item["output"]["finite"]),
        }
        for item in trace
    ]


class PILLARTrace:
    def __init__(self, model):
        self.trace = []
        self.handles = []
        for name, module in model.named_modules():
            if isinstance(module, PILLARPolynomialReLU):
                self.handles.append(module.register_forward_hook(
                    self._make_hook(name)))

    def _make_hook(self, name):
        def hook(module, inputs, output):
            del module
            self.trace.append({
                "name": name,
                "input": tensor_summary(inputs[0]),
                "output": tensor_summary(output),
            })
        return hook

    def close(self):
        for handle in self.handles:
            handle.remove()


def model_from_config(config_path, checkpoint, device):
    cfg = get_config(config_path)
    kwargs = {
        "dropout": 0.0,
        "fp16": False,
        "num_features": int(cfg.embedding_size),
        "pillar_approximation_range": float(getattr(
            cfg, "pillar_approximation_range", 5.0)),
        "pillar_regularization_range": float(getattr(
            cfg, "pillar_regularization_range", 4.8)),
        "pillar_regularization_exponent": int(getattr(
            cfg, "pillar_regularization_exponent", 10)),
        "pillar_training_clip": bool(getattr(
            cfg, "pillar_training_clip", True)),
        "pillar_penalty_reduction": str(getattr(
            cfg, "pillar_penalty_reduction", "mean")),
        "pillar_penalty_tail_cap": getattr(
            cfg, "pillar_penalty_tail_cap", None),
        "pillar_input_scale": float(getattr(
            cfg, "pillar_input_scale", 1.0)),
        "pillar_input_scale_overrides": dict(getattr(
            cfg, "pillar_input_scale_overrides", {})),
    }
    model = get_model(cfg.network, **kwargs)
    state = extract_state_dict(torch.load(checkpoint, map_location="cpu"))
    model.load_state_dict(state, strict=True)
    return model.float().to(device).eval()


@torch.no_grad()
def debug_rows(model, data, flip, start, end, batch_size, max_bad_rows):
    device = next(model.parameters()).device
    bad_rows = []
    bad_row_count = 0
    for batch_start in range(start, end, batch_size):
        batch_end = min(batch_start + batch_size, end)
        inputs = data[batch_start:batch_end].to(device=device)
        inputs = inputs / 127.5 - 1.0
        outputs = model(inputs)
        bad = (~torch.isfinite(outputs).all(dim=1)).nonzero().flatten()
        bad_row_count += int(bad.numel())
        remaining = max(0, max_bad_rows - len(bad_rows))
        for local_row in bad[:remaining].tolist():
            bad_rows.append(batch_start + local_row)

    print(json.dumps({
        "flip": flip,
        "bad_row_count": bad_row_count,
        "traced_bad_rows": bad_rows,
    }))
    for row in bad_rows:
        trace = PILLARTrace(model)
        try:
            value = data[row:row + 1].to(device=device)
            output = model(value / 127.5 - 1.0)
        finally:
            trace.close()
        first = first_nonfinite_activation(trace.trace)
        result = {
            "flip": flip,
            "row": row,
            "model_output": tensor_summary(output),
            "first_nonfinite_activation": (
                trace.trace[first]["name"] if first is not None else None),
            "activation": (
                trace.trace[first] if first is not None else None),
            "previous_activation": (
                trace.trace[first - 1]
                if first is not None and first > 0 else None),
            "activation_trace": compact_activation_trace(trace.trace),
        }
        print(json.dumps(result, sort_keys=True))
    return bad_rows


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--config", default="configs/ms1mv3_r50_pillar_espn_resume")
    parser.add_argument("--bin", required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-bad-rows", type=int, default=8)
    parser.add_argument("--flip", type=int, choices=(0, 1), default=0)
    parser.add_argument("--device", default="cuda:0")
    return parser


def main(args):
    model = model_from_config(
        args.config, args.checkpoint, torch.device(args.device))
    data_set = verification.load_bin(args.bin, (112, 112))
    data = data_set[0][args.flip]
    end = args.end if args.end > 0 else data.shape[0]
    if not 0 <= args.start < end <= data.shape[0]:
        raise ValueError(
            f"invalid row interval [{args.start}, {end}) for {data.shape[0]}")
    debug_rows(
        model, data, args.flip, args.start, end, args.batch_size,
        args.max_bad_rows)


if __name__ == "__main__":
    main(build_parser().parse_args())

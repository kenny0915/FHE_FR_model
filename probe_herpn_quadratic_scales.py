"""Probe safer quadratic coefficients on exact MS1Mv3 failure orientations.

The probe uses only training-set rows from a tail-mining manifest. It scales
one HerPN quadratic basis by increasing that basis BatchNorm's frozen
variance; the linear basis and every convolution remain unchanged. This is
an inference-time polynomial coefficient change and adds no FHE operation or
multiplicative depth.
"""

import argparse
import copy
import json
import os

import torch
from torch.utils.data import DataLoader, Subset

from backbones import get_model
from dataset import DatasetWithIndex, MXFaceDataset
from utils.utils_config import get_config
from utils.utils_tail_recovery import load_fixed_tail_replay_orientations


def attenuate_quadratic_basis_variance(variance, eps, scale):
    """Return a variance whose folded quadratic coefficient is ``scale``x."""
    scale = float(scale)
    if not 0.0 < scale <= 1.0:
        raise ValueError("quadratic scale must be in (0, 1]")
    if scale == 1.0:
        return variance.detach().clone()
    compute = variance.detach().to(dtype=torch.float64)
    adjusted = (compute + float(eps)) / (scale * scale) - float(eps)
    return adjusted.to(dtype=variance.dtype)


def atomic_json_dump(payload, path):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--manifest-key", default="output_nonfinite")
    parser.add_argument("--activation", action="append", required=True)
    parser.add_argument(
        "--quadratic-scale", action="append", type=float, required=True)
    parser.add_argument("--trace-activation", action="append", default=[])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--output", required=True)
    parser.add_argument("--save-first-zero-checkpoint")
    args = parser.parse_args()
    if args.batch_size <= 0 or args.workers < 0:
        raise ValueError("batch size must be positive and workers non-negative")

    cfg = get_config(args.config)
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)

    dataset = MXFaceDataset(cfg.rec, local_rank=0, range_augmentation=None)
    oriented_dataset = DatasetWithIndex(dataset, both_orientations=True)
    orientations = load_fixed_tail_replay_orientations(
        args.manifest, args.manifest_key)
    subset = Subset(
        oriented_dataset,
        [2 * source_index + orientation
         for source_index, orientation in orientations],
    )
    loader = DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=args.workers > 0,
    )

    state = torch.load(args.checkpoint, map_location="cpu")
    if isinstance(state, dict) and isinstance(
            state.get("state_dict_backbone"), dict):
        state = state["state_dict_backbone"]
    original_state = copy.deepcopy(state)
    activation_names = tuple(dict.fromkeys(args.activation))
    variance_keys = {
        name: f"{name}.herpn.bn2.running_var"
        for name in activation_names
    }
    missing_variances = [
        key for key in variance_keys.values() if key not in original_state]
    if missing_variances:
        raise ValueError(
            f"Checkpoint has no HerPN variances {missing_variances!r}")

    model = get_model(
        cfg.network,
        dropout=0.0,
        fp16=False,
        num_features=cfg.embedding_size,
        herpn_range_limit=float(getattr(cfg, "herpn_range_limit", 6.0)),
        herpn_bn_eps=float(getattr(cfg, "herpn_bn_eps", 1e-4)),
        herpn_progress=5.0,
    ).to(device)
    modules = dict(model.named_modules())
    missing_activations = [
        name for name in activation_names if name not in modules]
    if missing_activations:
        raise ValueError(f"Unknown activations {missing_activations!r}")
    missing = [name for name in args.trace_activation if name not in modules]
    if missing:
        raise ValueError(f"Unknown trace activations: {missing}")
    basis_eps = {
        name: float(modules[name].herpn.bn2.eps)
        for name in activation_names
    }

    results = []
    saved_scale = None
    for scale in args.quadratic_scale:
        candidate_state = copy.deepcopy(original_state)
        for name, variance_key in variance_keys.items():
            candidate_state[variance_key] = (
                attenuate_quadratic_basis_variance(
                    original_state[variance_key], basis_eps[name], scale))
        model.load_state_dict(candidate_state, strict=True)
        model.set_herpn_progress(5.0)
        model.eval()

        trace_absmax = {name: 0.0 for name in args.trace_activation}
        trace_nonfinite = {name: 0 for name in args.trace_activation}

        def make_hook(name):
            def capture(_, inputs):
                values = inputs[0].detach().float()
                finite = torch.isfinite(values)
                trace_nonfinite[name] += int(
                    (~finite).flatten(1).any(1).sum())
                finite_values = values[finite]
                if finite_values.numel():
                    trace_absmax[name] = max(
                        trace_absmax[name],
                        float(finite_values.abs().max().item()),
                    )
            return capture

        handles = [
            modules[name].register_forward_pre_hook(make_hook(name))
            for name in args.trace_activation
        ]
        output_nonfinite = []
        try:
            with torch.inference_mode():
                for images, _, indices, image_orientations in loader:
                    embeddings = model(images.to(device, non_blocking=True))
                    finite = torch.isfinite(embeddings).all(dim=1).cpu()
                    output_nonfinite.extend(
                        {
                            "source_index": int(index),
                            "orientation": int(orientation),
                        }
                        for index, orientation, is_finite in zip(
                            indices, image_orientations, finite)
                        if not bool(is_finite)
                    )
        finally:
            for handle in handles:
                handle.remove()

        result = {
            "quadratic_scale": float(scale),
            "output_nonfinite_count": len(output_nonfinite),
            "output_nonfinite": output_nonfinite,
            "trace_absmax": trace_absmax,
            "trace_nonfinite_input_count": trace_nonfinite,
        }
        results.append(result)
        print(json.dumps({
            key: value for key, value in result.items()
            if key != "output_nonfinite"
        }, sort_keys=True), flush=True)
        if (not output_nonfinite and saved_scale is None
                and args.save_first_zero_checkpoint):
            directory = os.path.dirname(args.save_first_zero_checkpoint)
            if directory:
                os.makedirs(directory, exist_ok=True)
            torch.save(candidate_state, args.save_first_zero_checkpoint)
            saved_scale = float(scale)

    atomic_json_dump({
        "format": "exact_ms1mv3_quadratic_scale_probe_v1",
        "checkpoint": args.checkpoint,
        "manifest": args.manifest,
        "manifest_key": args.manifest_key,
        "activations": list(activation_names),
        "orientation_count": len(orientations),
        "saved_zero_checkpoint": (
            args.save_first_zero_checkpoint if saved_scale is not None else None),
        "saved_zero_scale": saved_scale,
        "results": results,
    }, args.output)


if __name__ == "__main__":
    main()

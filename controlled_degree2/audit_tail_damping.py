"""Audit candidate c2 damping factors on exact MS1MV3 failure rows."""

from __future__ import annotations

import argparse
import json
import os

import torch
from torch.utils.data import DataLoader, Subset

from controlled_degree2.mine_deployment_tails import atomic_json_dump
from controlled_degree2.model import (
    load_controlled_checkpoint,
    quadratic_modules,
    set_quadratic_schedule,
)
from utils.utils_tail_recovery import load_fixed_tail_replay_orientations


def matches_scope(name, scopes):
    return any(name == scope or name.startswith(scope + ".") for scope in scopes)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--manifest-key", default="output_nonfinite")
    parser.add_argument("--output", required=True)
    parser.add_argument("--scope", action="append", required=True)
    parser.add_argument("--candidate", action="append", type=float, required=True)
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()
    if args.workers < 0:
        raise ValueError("workers must be non-negative")
    if any(not 0.0 < candidate <= 1.0 for candidate in args.candidate):
        raise ValueError("candidate damping factors must be in (0, 1]")

    from dataset import DatasetWithIndex, MXFaceDataset

    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    model, payload = load_controlled_checkpoint(args.checkpoint, device=device)
    model.to(memory_format=torch.channels_last).eval()
    set_quadratic_schedule(model, alpha=1.0, clip_eval=False)
    modules = list(quadratic_modules(model))
    selected = [module for module in modules if matches_scope(module.name, args.scope)]
    if not selected:
        raise ValueError(f"damping scopes match no activations: {args.scope}")
    baseline_c2 = {
        module.name: module.coeffs[:, 2].detach().clone() for module in modules
    }

    rows = load_fixed_tail_replay_orientations(
        args.manifest, key=args.manifest_key
    )
    dataset = MXFaceDataset(args.dataset_root, local_rank=0)
    bad = [row for row in rows if row[0] >= len(dataset)]
    if bad:
        raise ValueError(f"manifest indices exceed MS1MV3: {bad[:5]}")
    oriented = DatasetWithIndex(dataset, both_orientations=True)
    subset = Subset(oriented, [2 * index + orientation for index, orientation in rows])
    loader = DataLoader(
        subset,
        batch_size=len(subset),
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )
    images, _labels, indices, orientations = next(iter(loader))
    images = images.to(device, non_blocking=True).contiguous(
        memory_format=torch.channels_last
    )
    keys = list(zip(indices.tolist(), orientations.tolist()))

    current_max = None
    first_escape = None

    def make_hook(module):
        def capture(_module, inputs):
            nonlocal current_max, first_escape
            values = inputs[0].detach().float()
            lam = module.lam_reg.reshape(
                (1, module.channels) + (1,) * (values.ndim - 2)
            )
            ratio = torch.nan_to_num(
                values.abs() / lam,
                nan=float("inf"),
                posinf=float("inf"),
                neginf=float("inf"),
            ).flatten(1).amax(dim=1)
            current_max = torch.maximum(current_max, ratio)
            escaped = (ratio > 1.0).detach().cpu().tolist()
            ratios = ratio.detach().cpu().tolist()
            for position, hit in enumerate(escaped):
                if hit and first_escape[position] is None:
                    first_escape[position] = {
                        "module": module.name,
                        "ratio": float(ratios[position]),
                    }
        return capture

    handles = [module.register_forward_pre_hook(make_hook(module)) for module in modules]
    results = []
    try:
        with torch.inference_mode():
            for candidate in args.candidate:
                for module in modules:
                    scale = candidate if module in selected else 1.0
                    module.coeffs[:, 2].copy_(baseline_c2[module.name] * scale)
                current_max = torch.zeros(len(keys), device=device)
                first_escape = [None] * len(keys)
                embeddings = model(images)
                finite = torch.isfinite(embeddings).all(dim=1).cpu().tolist()
                maxima = current_max.cpu().tolist()
                candidate_rows = [
                    {
                        "source_index": key[0],
                        "orientation": key[1],
                        "output_finite": bool(is_finite),
                        "max_ratio": float(maximum),
                        "first_escape": escape,
                    }
                    for key, is_finite, maximum, escape in zip(
                        keys, finite, maxima, first_escape
                    )
                ]
                results.append({
                    "factor": float(candidate),
                    "nonfinite_outputs": sum(not value for value in finite),
                    "max_ratio": float(max(maxima)),
                    "rows": candidate_rows,
                })
                print(
                    f"factor={candidate:.6g} nonfinite="
                    f"{results[-1]['nonfinite_outputs']}/{len(keys)} "
                    f"max_ratio={results[-1]['max_ratio']:.6g}",
                    flush=True,
                )
    finally:
        for handle in handles:
            handle.remove()

    atomic_json_dump({
        "format": "controlled-degree2-ms1mv3-c2-damping-audit-v1",
        "checkpoint": os.path.abspath(args.checkpoint),
        "checkpoint_degree": int(payload["degree"]),
        "dataset": "MS1MV3",
        "dataset_root": os.path.abspath(args.dataset_root),
        "manifest": os.path.abspath(args.manifest),
        "manifest_key": args.manifest_key,
        "scopes": args.scope,
        "damped_activations": [module.name for module in selected],
        "results": results,
    }, args.output)


if __name__ == "__main__":
    main()

"""Locate source-only activation channels that seed numerical tail failures."""

from __future__ import annotations

import argparse
import json
import os

import torch
from torch.utils.data import DataLoader, Subset

from controlled_degree2.deployment_data import (
    build_deployment_dataset,
    parse_context_scales,
    parse_stress_variants,
)
from controlled_degree2.mine_deployment_tails import atomic_json_dump
from controlled_degree2.model import (
    load_controlled_checkpoint,
    quadratic_modules,
    set_quadratic_schedule,
)
from utils.utils_tail_recovery import load_fixed_tail_replay_orientations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument(
        "--dataset-type", choices=("ms1mv3", "wider", "ytf"), default="ytf"
    )
    parser.add_argument("--annotations", default=None)
    parser.add_argument("--wider-context-scales", default="1.0")
    parser.add_argument("--stress-variants", default="base")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--manifest-key", default="output_nonfinite")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.batch_size <= 0 or args.workers < 0:
        raise ValueError("batch size must be positive and workers non-negative")

    from dataset import DatasetWithIndex

    variants = parse_stress_variants(args.stress_variants)
    dataset = build_deployment_dataset(
        args.dataset_type,
        args.dataset_root,
        local_rank=0,
        annotations=args.annotations,
        wider_context_scales=parse_context_scales(args.wider_context_scales),
        wider_stress_variants=variants,
        aligned_stress_variants=variants,
    )
    with open(args.manifest, encoding="utf-8") as handle:
        manifest_payload = json.load(handle)
    manifest_dataset = manifest_payload.get("dataset")
    if manifest_dataset and manifest_dataset.lower() != args.dataset_type:
        raise ValueError("manifest dataset does not match requested dataset")
    manifest_digest = manifest_payload.get("dataset_index_digest")
    if (
        manifest_digest
        and manifest_digest != getattr(dataset, "index_digest", None)
    ):
        raise ValueError("dataset ordering differs from the mining manifest")
    rows = load_fixed_tail_replay_orientations(
        args.manifest, key=args.manifest_key
    )
    if any(index >= len(dataset) for index, _orientation in rows):
        raise ValueError("manifest indices exceed dataset")
    oriented = DatasetWithIndex(dataset, both_orientations=True)
    loader = DataLoader(
        Subset(
            oriented,
            [2 * index + orientation for index, orientation in rows],
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )

    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    model, payload = load_controlled_checkpoint(args.checkpoint, device=device)
    model.to(memory_format=torch.channels_last).eval()
    set_quadratic_schedule(model, alpha=1.0, clip_eval=False)
    modules = list(quadratic_modules(model))
    maxima = {
        module.name: torch.zeros(module.channels, device=device) for module in modules
    }
    escape_rows = {
        module.name: torch.zeros(
            module.channels, dtype=torch.int64, device=device
        )
        for module in modules
    }
    nonfinite_rows = {
        module.name: torch.zeros(
            module.channels, dtype=torch.int64, device=device
        )
        for module in modules
    }

    def make_hook(module):
        def capture(_module, inputs):
            values = inputs[0].detach().float()
            lam = module.lam_reg.reshape(
                (1, module.channels) + (1,) * (values.ndim - 2)
            )
            finite = torch.isfinite(values).flatten(2).all(dim=2)
            ratios = torch.nan_to_num(
                values.abs() / lam,
                nan=float("inf"),
                posinf=float("inf"),
                neginf=float("inf"),
            ).flatten(2).amax(dim=2)
            maxima[module.name] = torch.maximum(
                maxima[module.name], ratios.amax(dim=0)
            )
            escape_rows[module.name] += (ratios > 1.0).sum(dim=0)
            nonfinite_rows[module.name] += (~finite).sum(dim=0)

        return capture

    handles = [module.register_forward_pre_hook(make_hook(module)) for module in modules]
    output_nonfinite = 0
    try:
        with torch.inference_mode():
            for images, _labels, _indices, _orientations in loader:
                images = images.to(device, non_blocking=True).contiguous(
                    memory_format=torch.channels_last
                )
                embeddings = model(images)
                output_nonfinite += int(
                    (~torch.isfinite(embeddings).all(dim=1)).sum()
                )
    finally:
        for handle in handles:
            handle.remove()

    activations = {}
    for module in modules:
        channel_maxima = maxima[module.name].cpu().tolist()
        channel_escapes = escape_rows[module.name].cpu().tolist()
        channel_nonfinite = nonfinite_rows[module.name].cpu().tolist()
        activations[module.name] = {
            "channels": [
                {
                    "channel": channel,
                    "max_ratio": float(channel_maxima[channel]),
                    "escape_rows": int(channel_escapes[channel]),
                    "nonfinite_rows": int(channel_nonfinite[channel]),
                }
                for channel in sorted(
                    range(module.channels),
                    key=lambda index: (-channel_maxima[index], index),
                )
            ]
        }

    atomic_json_dump({
        "format": "controlled-degree2-source-tail-channels-v1",
        "checkpoint": os.path.abspath(args.checkpoint),
        "checkpoint_degree": int(payload["degree"]),
        "dataset": args.dataset_type,
        "dataset_root": os.path.abspath(args.dataset_root),
        "stress_variants": list(variants),
        "dataset_index_digest": getattr(dataset, "index_digest", None),
        "manifest": os.path.abspath(args.manifest),
        "manifest_key": args.manifest_key,
        "rows": len(rows),
        "output_nonfinite": output_nonfinite,
        "activation_names": [module.name for module in modules],
        "activations": activations,
    }, args.output)
    print(
        f"saved {os.path.abspath(args.output)}: rows={len(rows)} "
        f"output_nonfinite={output_nonfinite}",
        flush=True,
    )


if __name__ == "__main__":
    main()

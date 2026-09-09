"""Measure source-only embedding drift between controlled checkpoints."""

from __future__ import annotations

import argparse
import os

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from controlled_degree2.deployment_data import (
    build_deployment_dataset,
    parse_context_scales,
    parse_stress_variants,
)
from controlled_degree2.mine_deployment_tails import atomic_json_dump
from controlled_degree2.model import (
    load_controlled_checkpoint,
    set_quadratic_schedule,
)


def evenly_spaced_source_indices(length: int, count: int) -> tuple[int, ...]:
    """Return deterministic stratum centers without reading dataset labels."""
    if length <= 0 or count <= 0:
        raise ValueError("dataset length and sample count must be positive")
    count = min(length, count)
    return tuple(((2 * index + 1) * length) // (2 * count) for index in range(count))


def distribution(values: torch.Tensor) -> dict[str, float]:
    values = values.float()
    quantiles = torch.tensor((0.001, 0.01, 0.05, 0.5), dtype=torch.float32)
    q001, q01, q05, median = torch.quantile(values, quantiles).tolist()
    return {
        "mean": float(values.mean()),
        "minimum": float(values.min()),
        "p001": float(q001),
        "p01": float(q01),
        "p05": float(q05),
        "median": float(median),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument(
        "--dataset-type", choices=("ms1mv3", "wider", "ytf"), default="ytf"
    )
    parser.add_argument("--annotations", default=None)
    parser.add_argument("--wider-context-scales", default="1.0")
    parser.add_argument("--stress-variants", default="base")
    parser.add_argument("--source-samples", type=int, default=32768)
    parser.add_argument(
        "--both-orientations", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.source_samples <= 0 or args.batch_size <= 0 or args.workers < 0:
        raise ValueError("sample/batch sizes must be positive and workers non-negative")

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
    source_indices = evenly_spaced_source_indices(len(dataset), args.source_samples)
    oriented = DatasetWithIndex(
        dataset, both_orientations=args.both_orientations
    )
    if args.both_orientations:
        rows = [2 * index + orientation for index in source_indices for orientation in (0, 1)]
    else:
        rows = list(source_indices)
    loader = DataLoader(
        Subset(oriented, rows),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )

    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    reference, reference_payload = load_controlled_checkpoint(
        args.reference, device=device
    )
    candidate, candidate_payload = load_controlled_checkpoint(
        args.candidate, device=device
    )
    for model in (reference, candidate):
        model.to(memory_format=torch.channels_last).eval()
        set_quadratic_schedule(model, alpha=1.0, clip_eval=False)

    cosines = []
    norm_ratios = []
    reference_nonfinite = 0
    candidate_nonfinite = 0
    jointly_finite = 0
    with torch.inference_mode():
        for images, _labels, _indices, _orientations in loader:
            images = images.to(device, non_blocking=True).contiguous(
                memory_format=torch.channels_last
            )
            reference_embeddings = reference(images).float()
            candidate_embeddings = candidate(images).float()
            reference_finite = torch.isfinite(reference_embeddings).all(dim=1)
            candidate_finite = torch.isfinite(candidate_embeddings).all(dim=1)
            finite = reference_finite & candidate_finite
            reference_nonfinite += int((~reference_finite).sum())
            candidate_nonfinite += int((~candidate_finite).sum())
            jointly_finite += int(finite.sum())
            if finite.any():
                ref = reference_embeddings[finite]
                cand = candidate_embeddings[finite]
                cosines.append(F.cosine_similarity(cand, ref, dim=1).cpu())
                norm_ratios.append(
                    (cand.norm(dim=1) / ref.norm(dim=1).clamp_min(1e-12)).cpu()
                )

    if not cosines:
        raise FloatingPointError("no jointly finite embeddings for fidelity audit")
    cosine_values = torch.cat(cosines)
    norm_ratio_values = torch.cat(norm_ratios)
    result = {
        "format": "controlled-degree2-source-embedding-fidelity-v1",
        "reference": os.path.abspath(args.reference),
        "candidate": os.path.abspath(args.candidate),
        "reference_degree": int(reference_payload["degree"]),
        "candidate_degree": int(candidate_payload["degree"]),
        "dataset": args.dataset_type,
        "dataset_root": os.path.abspath(args.dataset_root),
        "stress_variants": list(variants),
        "dataset_index_digest": getattr(dataset, "index_digest", None),
        "dataset_source_rows": len(dataset),
        "sampled_source_rows": len(source_indices),
        "both_orientations": bool(args.both_orientations),
        "inference_rows": len(rows),
        "jointly_finite_rows": jointly_finite,
        "reference_nonfinite_rows": reference_nonfinite,
        "candidate_nonfinite_rows": candidate_nonfinite,
        "cosine_similarity": distribution(cosine_values),
        "embedding_norm_ratio": distribution(norm_ratio_values),
    }
    atomic_json_dump(result, args.output)
    print(
        f"saved {os.path.abspath(args.output)}: rows={len(rows)} "
        f"finite={jointly_finite} cosine_mean="
        f"{result['cosine_similarity']['mean']:.8f} "
        f"cosine_p01={result['cosine_similarity']['p01']:.8f}",
        flush=True,
    )


if __name__ == "__main__":
    main()

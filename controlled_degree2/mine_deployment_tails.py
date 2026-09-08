"""Mine rare non-IJB tails through the real unclipped degree-2 graph.

This is a read-only, evaluation-mode scanner.  It never reads IJB data and
does not modify the checkpoint. The output records deterministic source and
orientation rows with the largest per-layer input-to-interval ratios so that
training can replay them through an unclipped deployment shadow path.
"""

from __future__ import annotations

import argparse
import heapq
import json
import os

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from controlled_degree2.model import (
    load_controlled_checkpoint,
    quadratic_modules,
    set_quadratic_schedule,
)
from controlled_degree2.deployment_data import (
    build_deployment_dataset,
    parse_context_scales,
    parse_stress_variants,
)


def atomic_json_dump(payload, path):
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def update_tail_heap(heap, ratios, magnitudes, source_indices, orientations, topk):
    """Keep the largest finite normalized tails for one activation."""
    if topk <= 0 or ratios.numel() == 0:
        return
    count = min(int(topk), int(ratios.numel()))
    values, positions = ratios.topk(count, sorted=False)
    values = values.detach().cpu().tolist()
    positions = positions.detach().cpu().tolist()
    magnitudes = magnitudes.detach().cpu().tolist()
    for value, position in zip(values, positions):
        item = (
            float(value),
            int(source_indices[position]),
            int(orientations[position]),
            float(magnitudes[position]),
        )
        if len(heap) < topk:
            heapq.heappush(heap, item)
        elif item > heap[0]:
            heapq.heapreplace(heap, item)


def merge_rank_payloads(payloads, activation_names, topk):
    """Deduplicate rank shards and build a layer-balanced replay ordering."""
    merged = {name: {} for name in activation_names}
    output_nonfinite = set()
    for payload in payloads:
        output_nonfinite.update(
            (int(row["source_index"]), int(row["orientation"]))
            for row in payload["output_nonfinite"]
        )
        for name in activation_names:
            for row in payload["activations"][name]["tail"]:
                key = (int(row["source_index"]), int(row["orientation"]))
                candidate = (float(row["ratio"]), float(row["absmax"]))
                if candidate > merged[name].get(key, (float("-inf"), 0.0)):
                    merged[name][key] = candidate

    ordered_by_activation = {}
    activations = {}
    for name in activation_names:
        ordered = sorted(
            merged[name].items(),
            key=lambda item: (-item[1][0], item[0]),
        )[:topk]
        ordered_by_activation[name] = ordered
        activations[name] = {
            "tail": [
                {
                    "source_index": key[0],
                    "orientation": key[1],
                    "ratio": values[0],
                    "absmax": values[1],
                }
                for key, values in ordered
            ],
            "nonfinite_input_count": sum(
                int(payload["activations"][name]["nonfinite_input_count"])
                for payload in payloads
            ),
        }

    # Exact MS1MV3 failures have first priority.  Then round-robin normalized
    # tails across layers so a numerically larger stage cannot monopolize the
    # replay set merely because its activation units differ.
    combined = []
    seen = set()
    for key in sorted(output_nonfinite):
        if key not in seen:
            seen.add(key)
            combined.append({"source_index": key[0], "orientation": key[1]})
    for position in range(topk):
        for name in activation_names:
            ordered = ordered_by_activation[name]
            if position >= len(ordered):
                continue
            key = ordered[position][0]
            if key not in seen:
                seen.add(key)
                combined.append({"source_index": key[0], "orientation": key[1]})

    return {
        "format": "controlled-degree2-deployment-tails-v2",
        "selection": "unclipped eval graph, normalized by per-channel lam_reg",
        "activation_names": list(activation_names),
        "topk": int(topk),
        "combined_orientations": combined,
        "combined_source_indices": list(dict.fromkeys(
            row["source_index"] for row in combined
        )),
        "output_nonfinite": [
            {"source_index": key[0], "orientation": key[1]}
            for key in sorted(output_nonfinite)
        ],
        "activations": activations,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument(
        "--dataset-type", choices=("ms1mv3", "wider", "ytf"), default="ms1mv3"
    )
    parser.add_argument("--annotations", default=None)
    parser.add_argument("--wider-context-scales", default="1.0,1.5")
    parser.add_argument("--wider-stress-variants", default="base")
    parser.add_argument("--aligned-stress-variants", default="base")
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--topk", type=int, default=64)
    parser.add_argument(
        "--both-orientations", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--progress-batches", type=int, default=100)
    parser.add_argument("--limit-batches", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.batch_size <= 0 or args.workers < 0 or args.topk <= 0:
        raise ValueError("batch-size/topk must be positive and workers non-negative")
    if args.limit_batches < 0 or args.progress_batches < 0:
        raise ValueError("batch limits/progress interval must be non-negative")

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    from dataset import DatasetWithIndex

    context_scales = parse_context_scales(args.wider_context_scales)
    stress_variants = parse_stress_variants(args.wider_stress_variants)
    aligned_stress_variants = parse_stress_variants(args.aligned_stress_variants)
    base_dataset = build_deployment_dataset(
        args.dataset_type,
        args.dataset_root,
        local_rank=local_rank,
        annotations=args.annotations,
        wider_context_scales=context_scales,
        wider_stress_variants=stress_variants,
        aligned_stress_variants=aligned_stress_variants,
    )
    indexed_dataset = DatasetWithIndex(
        base_dataset, both_orientations=args.both_orientations
    )
    sampler = DistributedSampler(
        indexed_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        drop_last=False,
    )
    loader = DataLoader(
        indexed_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=args.workers > 0,
        prefetch_factor=2 if args.workers > 0 else None,
    )

    model, checkpoint_payload = load_controlled_checkpoint(
        args.checkpoint, device=device
    )
    model.to(memory_format=torch.channels_last)
    model.eval()
    set_quadratic_schedule(model, alpha=1.0, clip_eval=False)
    modules = list(quadratic_modules(model))
    activation_names = tuple(module.name for module in modules)
    heaps = {name: [] for name in activation_names}
    nonfinite_input_counts = {name: 0 for name in activation_names}
    output_nonfinite = set()
    current_indices = []
    current_orientations = []

    def make_hook(module):
        def capture(_module, inputs):
            values = inputs[0].detach().float()
            flat = values.flatten(1)
            finite = torch.isfinite(flat).all(dim=1)
            nonfinite_input_counts[module.name] += int((~finite).sum())
            lam = module.lam_reg.reshape(
                (1, module.channels) + (1,) * (values.ndim - 2)
            )
            ratios = torch.nan_to_num(
                values.abs() / lam,
                nan=float("inf"),
                posinf=float("inf"),
                neginf=float("inf"),
            ).flatten(1).amax(dim=1)
            magnitudes = torch.nan_to_num(
                values.abs(),
                nan=float("inf"),
                posinf=float("inf"),
                neginf=float("inf"),
            ).flatten(1).amax(dim=1)
            positions = finite.nonzero(as_tuple=False).flatten()
            if positions.numel():
                cpu_positions = positions.detach().cpu().tolist()
                update_tail_heap(
                    heaps[module.name],
                    ratios[positions],
                    magnitudes[positions],
                    [current_indices[position] for position in cpu_positions],
                    [current_orientations[position] for position in cpu_positions],
                    args.topk,
                )
        return capture

    handles = [module.register_forward_pre_hook(make_hook(module)) for module in modules]
    completed = 0
    try:
        with torch.inference_mode():
            for images, _labels, indices, orientations in loader:
                if args.limit_batches and completed >= args.limit_batches:
                    break
                current_indices = indices.tolist()
                current_orientations = orientations.tolist()
                images = images.to(device, non_blocking=True).contiguous(
                    memory_format=torch.channels_last
                )
                embeddings = model(images)
                finite = torch.isfinite(embeddings).all(dim=1).cpu().tolist()
                output_nonfinite.update(
                    (int(index), int(orientation))
                    for index, orientation, is_finite in zip(
                        current_indices, current_orientations, finite
                    )
                    if not is_finite
                )
                completed += 1
                if (
                    rank == 0
                    and args.progress_batches
                    and completed % args.progress_batches == 0
                ):
                    print(
                        f"deployment mining batches/rank={completed}/{len(loader)}",
                        flush=True,
                    )
    finally:
        for handle in handles:
            handle.remove()

    shard = {
        "rank": rank,
        "batches": completed,
        "output_nonfinite": [
            {"source_index": key[0], "orientation": key[1]}
            for key in sorted(output_nonfinite)
        ],
        "activations": {
            name: {
                "nonfinite_input_count": nonfinite_input_counts[name],
                "tail": [
                    {
                        "source_index": source_index,
                        "orientation": orientation,
                        "ratio": ratio,
                        "absmax": magnitude,
                    }
                    for ratio, source_index, orientation, magnitude in sorted(
                        heaps[name], reverse=True
                    )
                ],
            }
            for name in activation_names
        },
    }
    shard_path = f"{args.output}.rank{rank}.json"
    atomic_json_dump(shard, shard_path)
    if world_size > 1:
        dist.barrier()

    if rank == 0:
        payloads = []
        for shard_rank in range(world_size):
            with open(
                f"{args.output}.rank{shard_rank}.json", encoding="utf-8"
            ) as handle:
                payloads.append(json.load(handle))
        merged = merge_rank_payloads(payloads, activation_names, args.topk)
        merged.update({
            "checkpoint": os.path.abspath(args.checkpoint),
            "checkpoint_degree": int(checkpoint_payload["degree"]),
            "dataset": args.dataset_type,
            "dataset_root": os.path.abspath(args.dataset_root),
            "dataset_source_rows": len(base_dataset),
            "annotations": (
                os.path.abspath(args.annotations) if args.annotations else None
            ),
            "wider_context_scales": (
                list(context_scales) if args.dataset_type == "wider" else None
            ),
            "wider_stress_variants": (
                list(stress_variants) if args.dataset_type == "wider" else None
            ),
            "aligned_stress_variants": (
                list(aligned_stress_variants)
                if args.dataset_type == "ytf" else None
            ),
            "dataset_index_digest": getattr(base_dataset, "index_digest", None),
            "both_orientations": bool(args.both_orientations),
            "world_size": world_size,
            "batches_per_rank": [payload["batches"] for payload in payloads],
        })
        atomic_json_dump(merged, args.output)
        print(
            f"saved {args.output}: replay_rows="
            f"{len(merged['combined_orientations'])} output_nonfinite="
            f"{len(merged['output_nonfinite'])}",
            flush=True,
        )
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

"""Mine exact MS1Mv3 activation tails for a fully polynomial R50.

This scanner never changes model state.  It evaluates the unclipped epoch-23
graph in eval mode, records per-image pre-HerPN maxima, and writes stable
MS1Mv3 source indices for later training replay.  IJB-C is not read here.
"""

import argparse
import heapq
import json
import os
from collections import defaultdict

import torch
from torch import distributed
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from backbones import get_model
from dataset import DatasetWithIndex, get_dataloader
from utils.utils_config import get_config


def atomic_json_dump(payload, path):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def update_tail_heap(heap, magnitudes, source_indices, orientations, topk):
    """Keep the strongest unique source/orientation observations."""
    if topk <= 0 or magnitudes.numel() == 0:
        return
    count = min(int(topk), int(magnitudes.numel()))
    values, positions = magnitudes.topk(count, sorted=False)
    for value, position in zip(
            values.detach().cpu().tolist(), positions.detach().cpu().tolist()):
        item = (
            float(value),
            int(source_indices[int(position)]),
            int(orientations[int(position)]),
        )
        if len(heap) < topk:
            heapq.heappush(heap, item)
        elif item > heap[0]:
            heapq.heapreplace(heap, item)


def merge_rank_payloads(payloads, activation_names, topk):
    merged = {name: {} for name in activation_names}
    output_nonfinite = set()
    for payload in payloads:
        output_nonfinite.update(
            (int(item["source_index"]), int(item["orientation"]))
            for item in payload["output_nonfinite"])
        for name in activation_names:
            for item in payload["activations"][name]["tail"]:
                key = (int(item["source_index"]), int(item["orientation"]))
                merged[name][key] = max(
                    float(item["absmax"]),
                    merged[name].get(key, float("-inf")),
                )

    activation_payload = {}
    ordered_by_activation = {}
    for name in activation_names:
        ordered = sorted(
            merged[name].items(), key=lambda item: (-item[1], item[0]))[:topk]
        ordered_by_activation[name] = ordered
        activation_payload[name] = {
            "tail": [
                {
                    "source_index": source_index,
                    "orientation": orientation,
                    "absmax": magnitude,
                }
                for (source_index, orientation), magnitude in ordered
            ],
            "nonfinite_input_count": sum(
                int(payload["activations"][name]["nonfinite_input_count"])
                for payload in payloads),
        }

    # Round-robin prevents the numerically largest stage from monopolizing
    # replay solely because magnitudes at different depths use different units.
    # Exact non-finite training sources are the strongest available evidence
    # and must occupy the replay priority prefix before merely large finite
    # tails.  Orientation is intentionally collapsed because the training
    # transform resamples horizontal flips on every replay visit.
    combined = []
    seen = set()
    for source_index, _ in sorted(output_nonfinite):
        if source_index not in seen:
            seen.add(source_index)
            combined.append(source_index)
    for position in range(topk):
        for name in activation_names:
            ordered = ordered_by_activation[name]
            if position >= len(ordered):
                continue
            source_index = ordered[position][0][0]
            if source_index not in seen:
                seen.add(source_index)
                combined.append(source_index)

    return {
        "format": "exact_ms1mv3_herpn_tail_v1",
        "activation_names": list(activation_names),
        "topk_per_rank": int(topk),
        "exact_nonfinite_source_count": len({
            source_index for source_index, _ in output_nonfinite}),
        "combined_source_indices": combined,
        "output_nonfinite": [
            {"source_index": source_index, "orientation": orientation}
            for source_index, orientation in sorted(output_nonfinite)
        ],
        "activations": activation_payload,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--activation", action="append", required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--topk", type=int, default=512)
    parser.add_argument("--both-orientations", action="store_true")
    parser.add_argument("--progress-batches", type=int, default=100)
    args = parser.parse_args()
    if args.batch_size <= 0 or args.workers < 0 or args.topk <= 0:
        raise ValueError("batch-size/topk must be positive and workers non-negative")

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if not distributed.is_initialized():
        distributed.init_process_group(
            "nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    cfg = get_config(args.config)
    base_loader = get_dataloader(
        cfg.rec, local_rank, args.batch_size, False, False,
        cfg.seed, args.workers, drop_last=False, range_augmentation=None)
    indexed_dataset = DatasetWithIndex(
        base_loader.dataset, both_orientations=args.both_orientations)
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
        worker_init_fn=base_loader.worker_init_fn,
    )

    model = get_model(
        cfg.network,
        dropout=0.0,
        fp16=False,
        num_features=cfg.embedding_size,
        herpn_range_limit=float(getattr(cfg, "herpn_range_limit", 6.0)),
        herpn_bn_eps=float(getattr(cfg, "herpn_bn_eps", 1e-4)),
        herpn_progress=5.0,
    ).to(device)
    state = torch.load(args.checkpoint, map_location="cpu")
    if isinstance(state, dict) and isinstance(state.get("state_dict_backbone"), dict):
        state = state["state_dict_backbone"]
    model.load_state_dict(state, strict=True)
    model.set_herpn_progress(5.0)
    model.eval()

    modules = dict(model.named_modules())
    activation_names = tuple(args.activation)
    missing = [name for name in activation_names if name not in modules]
    if missing:
        raise ValueError(f"Unknown activation names: {missing}")

    heaps = {name: [] for name in activation_names}
    nonfinite_input_counts = defaultdict(int)
    output_nonfinite = set()
    current_indices = []
    current_orientations = []

    def make_hook(name):
        def capture(_, inputs):
            values = inputs[0].detach().float()
            finite = torch.isfinite(values).flatten(1).all(dim=1)
            nonfinite_input_counts[name] += int((~finite).sum().item())
            magnitudes = torch.nan_to_num(
                values.abs(), nan=float("inf"),
                posinf=float("inf"), neginf=float("inf")).flatten(1).amax(dim=1)
            finite_positions = finite.nonzero(as_tuple=False).flatten()
            if finite_positions.numel() > 0:
                positions = finite_positions.detach().cpu().tolist()
                update_tail_heap(
                    heaps[name], magnitudes[finite_positions],
                    [current_indices[position] for position in positions],
                    [current_orientations[position] for position in positions],
                    args.topk)
        return capture

    handles = [
        modules[name].register_forward_pre_hook(make_hook(name))
        for name in activation_names
    ]
    completed = 0
    try:
        with torch.inference_mode():
            for images, _, indices, orientations in loader:
                current_indices = indices.tolist()
                current_orientations = orientations.tolist()
                embeddings = model(images.to(device, non_blocking=True))
                finite = torch.isfinite(embeddings).all(dim=1).cpu().tolist()
                output_nonfinite.update(
                    (int(index), int(orientation))
                    for index, orientation, is_finite in zip(
                        current_indices, current_orientations, finite)
                    if not is_finite
                )
                completed += 1
                if (rank == 0 and args.progress_batches > 0
                        and completed % args.progress_batches == 0):
                    print(
                        f"mining batches/rank={completed}/{len(loader)}",
                        flush=True)
    finally:
        for handle in handles:
            handle.remove()

    shard_payload = {
        "rank": rank,
        "batches": completed,
        "output_nonfinite": [
            {"source_index": index, "orientation": orientation}
            for index, orientation in sorted(output_nonfinite)
        ],
        "activations": {
            name: {
                "nonfinite_input_count": nonfinite_input_counts[name],
                "tail": [
                    {
                        "source_index": index,
                        "orientation": orientation,
                        "absmax": magnitude,
                    }
                    for magnitude, index, orientation in sorted(
                        heaps[name], reverse=True)
                ],
            }
            for name in activation_names
        },
    }
    shard_path = args.output + f".rank{rank}.json"
    atomic_json_dump(shard_payload, shard_path)
    distributed.barrier()

    if rank == 0:
        payloads = []
        for shard_rank in range(world_size):
            with open(
                    args.output + f".rank{shard_rank}.json",
                    encoding="utf-8") as handle:
                payloads.append(json.load(handle))
        merged = merge_rank_payloads(payloads, activation_names, args.topk)
        merged.update({
            "checkpoint": args.checkpoint,
            "dataset": cfg.rec,
            "both_orientations": bool(args.both_orientations),
            "world_size": world_size,
            "batches_per_rank": completed,
        })
        atomic_json_dump(merged, args.output)
        print(
            f"saved {len(merged['combined_source_indices'])} unique hard-tail "
            f"indices to {args.output}; exact output nonfinite="
            f"{len(merged['output_nonfinite'])}",
            flush=True,
        )
    distributed.barrier()
    distributed.destroy_process_group()


if __name__ == "__main__":
    main()

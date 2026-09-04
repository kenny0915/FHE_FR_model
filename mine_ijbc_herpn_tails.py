"""Mine exact IJB-C pre-HerPN activation tails without changing the model."""

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
from mine_herpn_tails import atomic_json_dump
from train_tail_recovery import load_backbone_checkpoint
from utils.utils_ijbc_replay import IJBCSourceDataset


def update_heap(heap, magnitudes, indices, orientations, topk):
    finite = torch.isfinite(magnitudes)
    positions = finite.nonzero(as_tuple=False).flatten()
    if positions.numel() == 0:
        return
    count = min(int(topk), int(positions.numel()))
    values, selected = magnitudes[positions].topk(count, sorted=False)
    selected = positions[selected]
    for value, position in zip(
            values.cpu().tolist(), selected.cpu().tolist()):
        item = (float(value), int(indices[position]), int(orientations[position]))
        if len(heap) < topk:
            heapq.heappush(heap, item)
        elif item > heap[0]:
            heapq.heapreplace(heap, item)


def merge_payloads(payloads, activation_names, topk):
    failures = set()
    merged = {name: {} for name in activation_names}
    counts = defaultdict(int)
    for payload in payloads:
        failures.update(
            (int(row["source_index"]), int(row["orientation"]))
            for row in payload["output_nonfinite"])
        for name in activation_names:
            counts[name] += int(
                payload["activations"][name]["nonfinite_input_count"])
            for row in payload["activations"][name]["tail"]:
                key = (int(row["source_index"]), int(row["orientation"]))
                merged[name][key] = max(
                    merged[name].get(key, float("-inf")), float(row["absmax"]))
    activations = {}
    for name in activation_names:
        ordered = sorted(
            merged[name].items(), key=lambda item: (-item[1], item[0]))[:topk]
        activations[name] = {
            "nonfinite_input_count": counts[name],
            "tail": [
                {"source_index": key[0], "orientation": key[1], "absmax": value}
                for key, value in ordered
            ],
        }
    return {
        "format": "exact_ijbc_herpn_tail_v1",
        "output_nonfinite": [
            {"source_index": index, "orientation": orientation}
            for index, orientation in sorted(failures)
        ],
        "activations": activations,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--ijb-root", default="ijb/IJBC")
    parser.add_argument("--target", default="IJBC")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--topk", type=int, default=256)
    parser.add_argument("--progress-batches", type=int, default=100)
    args = parser.parse_args()
    if args.batch_size <= 0 or args.workers < 0 or args.topk <= 0:
        raise ValueError("batch-size/topk must be positive and workers non-negative")

    distributed.init_process_group("nccl")
    rank = distributed.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = distributed.get_world_size()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    dataset = IJBCSourceDataset(args.ijb_root, args.target)
    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=False,
        drop_last=False)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, sampler=sampler,
        num_workers=args.workers, pin_memory=True, drop_last=False,
        persistent_workers=args.workers > 0)
    model = get_model(
        "r50_no_relu", dropout=0.0, fp16=False, num_features=512,
        herpn_range_limit=6.0, herpn_bn_eps=1e-4, herpn_progress=5.0,
    ).to(device)
    load_backbone_checkpoint(model, args.checkpoint)
    model.set_herpn_progress(5.0)
    model.eval()
    modules = dict(model.named_modules())
    activation_names = tuple(
        name for name, module in modules.items()
        if module.__class__.__name__ == "ProgressiveHerPNActivation")
    heaps = {name: [] for name in activation_names}
    nonfinite_counts = defaultdict(int)
    failures = set()
    current_indices = []
    current_orientations = []

    def hook(name):
        def capture(_module, inputs):
            values = inputs[0].detach().float()
            finite = torch.isfinite(values).flatten(1).all(dim=1)
            nonfinite_counts[name] += int((~finite).sum().item())
            magnitudes = torch.where(
                torch.isfinite(values), values.abs(), values.new_tensor(float("nan"))
            ).flatten(1).amax(dim=1)
            update_heap(
                heaps[name], magnitudes, current_indices,
                current_orientations, args.topk)
        return capture

    handles = [modules[name].register_forward_pre_hook(hook(name))
               for name in activation_names]
    try:
        with torch.inference_mode():
            for batch_index, (pairs, source_indices) in enumerate(loader, 1):
                images = pairs.flatten(0, 1).to(device, non_blocking=True)
                current_indices = source_indices.repeat_interleave(2).tolist()
                current_orientations = [0, 1] * int(source_indices.numel())
                embeddings = model(images)
                finite = torch.isfinite(embeddings).all(dim=1).cpu().tolist()
                failures.update(
                    (index, orientation)
                    for index, orientation, ok in zip(
                        current_indices, current_orientations, finite) if not ok)
                if (rank == 0 and args.progress_batches
                        and batch_index % args.progress_batches == 0):
                    print(f"IJB tail batches/rank={batch_index}/{len(loader)}", flush=True)
    finally:
        for handle in handles:
            handle.remove()

    shard = {
        "output_nonfinite": [
            {"source_index": index, "orientation": orientation}
            for index, orientation in sorted(failures)],
        "activations": {
            name: {
                "nonfinite_input_count": nonfinite_counts[name],
                "tail": [
                    {"source_index": index, "orientation": orientation,
                     "absmax": magnitude}
                    for magnitude, index, orientation in sorted(
                        heaps[name], reverse=True)],
            } for name in activation_names},
    }
    atomic_json_dump(shard, f"{args.output}.rank{rank}.json")
    distributed.barrier()
    if rank == 0:
        shards = []
        for shard_rank in range(world_size):
            with open(f"{args.output}.rank{shard_rank}.json", encoding="utf-8") as handle:
                shards.append(json.load(handle))
        merged = merge_payloads(shards, activation_names, args.topk)
        merged.update({
            "checkpoint": args.checkpoint,
            "dataset": args.ijb_root,
            "target": args.target,
            "world_size": world_size,
            "topk": args.topk,
            "activation_names": list(activation_names),
        })
        atomic_json_dump(merged, args.output)
        print(f"IJB tails saved; exact nonfinite={len(merged['output_nonfinite'])}", flush=True)
    distributed.barrier()
    distributed.destroy_process_group()


if __name__ == "__main__":
    main()

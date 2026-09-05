"""Strict distributed output and pre-HerPN range gate for checkpoints."""

import argparse
import os

import torch
from torch import distributed

from dataset import MXFaceDataset, PairedOrientationDataset
from mine_herpn_tails import atomic_json_dump
from train_ijbc_numerical_calibration import (
    build_model,
    full_dataset_gate,
    make_loader,
)
from utils.utils_config import get_config
from utils.utils_widerface import WIDERFaceDataset


def make_gate_dataset(cfg, dataset_name, local_rank, wider_split):
    dataset_name = str(dataset_name).lower()
    if dataset_name == "ms1mv3":
        base = MXFaceDataset(cfg.rec, local_rank)
    elif dataset_name == "wider":
        base = WIDERFaceDataset(
            cfg.wider_image_root, cfg.wider_annotation_path,
            split=str(wider_split),
            validation_modulo=int(cfg.wider_validation_modulo),
            validation_fold=int(cfg.wider_validation_fold),
            min_face_size=int(cfg.wider_min_face_size),
            crop_scale=float(cfg.wider_crop_scale),
        )
    else:
        raise ValueError("Numerical gate dataset must be ms1mv3 or wider")
    return PairedOrientationDataset(base)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--dataset", choices=("ms1mv3", "wider"), required=True)
    parser.add_argument("--wider-split", default="validation",
                        choices=("calibration", "validation", "all"))
    parser.add_argument("--range-gate-limit", type=float, default=4.0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    if args.range_gate_limit <= 0 or args.batch_size <= 0 or args.workers < 0:
        raise ValueError("Range limit/batch size must be positive and workers non-negative")

    distributed.init_process_group("nccl")
    rank = distributed.get_rank()
    world_size = distributed.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    cfg = get_config(args.config)
    dataset = make_gate_dataset(
        cfg, args.dataset, local_rank, args.wider_split)
    loader, sampler = make_loader(
        dataset, args.batch_size, args.workers,
        rank, world_size, False, int(cfg.seed))
    os.makedirs(args.output_dir, exist_ok=True)

    for checkpoint_index, checkpoint in enumerate(args.checkpoint):
        sampler.set_epoch(checkpoint_index)
        model = build_model(cfg, checkpoint, device, penalty_mode=False)
        activation_names = tuple(
            name for name, module in model.named_modules()
            if module.__class__.__name__ == "ProgressiveHerPNActivation")
        if len(activation_names) != 25:
            raise RuntimeError(
                f"Expected 25 HerPN activations, found {len(activation_names)}")
        gate = full_dataset_gate(
            model, loader, device, rank, world_size,
            activation_names=activation_names,
            range_gate_limit=float(args.range_gate_limit))
        output_failures = gate["output_nonfinite"]
        range_violations = gate["range_violations"]
        failures = set(output_failures)
        failures.update((row[0], row[1]) for row in range_violations)
        payload = {
            "format": f"exact_{args.dataset}_numerical_gate_v1",
            "dataset": args.dataset,
            "wider_split": args.wider_split if args.dataset == "wider" else None,
            "checkpoint": checkpoint,
            "total_augmented_embeddings": 2 * len(dataset),
            "nonfinite_count": len(output_failures),
            "range_gate_limit": float(args.range_gate_limit),
            "range_violation_count": len(range_violations),
            "numerical_failure_count": len(failures),
            "max_preherpn_absmax": gate["max_preherpn_absmax"],
            "output_nonfinite": [
                {"source_index": index, "orientation": orientation}
                for index, orientation in output_failures],
            "range_violations": [
                {
                    "source_index": index,
                    "orientation": orientation,
                    "first_activation": activation,
                    "input_absmax": peak,
                }
                for index, orientation, activation, peak in range_violations],
        }
        if rank == 0:
            stem = os.path.splitext(os.path.basename(checkpoint))[0]
            output = os.path.join(
                args.output_dir, f"{stem}_{args.dataset}_gate.json")
            atomic_json_dump(payload, output)
            print(
                f"{checkpoint}: nonfinite={len(output_failures)} "
                f"range={len(range_violations)} union={len(failures)}/"
                f"{2 * len(dataset)} max={gate['max_preherpn_absmax']}",
                flush=True)
        del model
        torch.cuda.empty_cache()
        distributed.barrier()
    distributed.destroy_process_group()


if __name__ == "__main__":
    main()

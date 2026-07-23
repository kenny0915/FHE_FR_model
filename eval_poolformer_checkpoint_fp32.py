#!/usr/bin/env python3
"""Evaluate a progressive SimpleGate PoolFormer checkpoint in full FP32."""

import argparse
import os
import re

import torch

from backbones import get_model
from utils.utils_config import get_config


def infer_completed_epoch(checkpoint_path):
    match = re.search(r"model_epoch_(\d+)\.pt$", os.path.basename(checkpoint_path))
    return int(match.group(1)) if match else None


def gate_blends_for_epoch(completed_epoch, group_epochs, transition_epochs):
    transition_epochs = float(transition_epochs)
    if transition_epochs <= 0:
        raise ValueError("simple_gate_transition_epochs must be positive")
    return tuple(
        min(max(
            (float(completed_epoch) - float(start)) / transition_epochs,
            0.0,
        ), 1.0)
        for start in group_epochs
    )


def extract_backbone_state(checkpoint):
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint must contain a state dictionary")
    for key in ("state_dict_backbone", "state_dict", "model"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            checkpoint = value
            break
    if checkpoint and all(key.startswith("module.") for key in checkpoint):
        checkpoint = {
            key[len("module."):]: value for key, value in checkpoint.items()
        }
    return checkpoint


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Load a progressive SimpleGate PoolFormer checkpoint into an FP32 "
            "model and run LFW/CFP-FP/AgeDB verification."
        )
    )
    parser.add_argument(
        "--config",
        default="configs/ms1mv3_poolformer_s24_no_ln_x2_act.py",
        help="Training config used to construct the checkpoint.",
    )
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path.")
    parser.add_argument(
        "--epoch",
        type=int,
        default=None,
        help=(
            "Completed epoch represented by model.pt or a full checkpoint. "
            "Automatically inferred from model_epoch_XX.pt."
        ),
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        default=None,
        help="Verification targets; defaults to config.val_targets.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--nfolds", type=int, default=10)
    return parser


def main():
    args = build_parser().parse_args()
    cfg = get_config(args.config)
    completed_epoch = (
        args.epoch
        if args.epoch is not None
        else infer_completed_epoch(args.checkpoint)
    )
    raw_checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if completed_epoch is None and isinstance(raw_checkpoint, dict):
        completed_epoch = raw_checkpoint.get("epoch")
    if completed_epoch is None:
        raise ValueError(
            "Cannot determine checkpoint epoch; pass --epoch explicitly")

    model = get_model(
        cfg.network,
        fp16=False,
        num_features=cfg.embedding_size,
        gate_range_limit=float(cfg.simple_gate_range_limit),
        gate_stats_sample_size=int(cfg.simple_gate_stats_sample_size),
        gate_compute_fp32=True,
        gate_fail_on_nonfinite=True,
        gate_initial_blend=0.0,
    )
    state = extract_backbone_state(raw_checkpoint)
    checkpoint_dtypes = sorted({
        str(value.dtype) for value in state.values()
        if torch.is_tensor(value) and value.is_floating_point()
    })
    model.load_state_dict(state, strict=True)

    group_epochs = tuple(cfg.simple_gate_group_epochs)
    blends = gate_blends_for_epoch(
        completed_epoch,
        group_epochs,
        cfg.simple_gate_transition_epochs,
    )
    model.set_simple_gate_blends(blends)
    for module in model.modules():
        if module.__class__.__name__ == "RepBatchNorm2d":
            module.set_progress(1, 1)

    device = torch.device(args.device)
    model.float().to(device).eval()
    parameter_dtypes = {
        parameter.dtype for parameter in model.parameters()
        if parameter.is_floating_point()
    }
    if parameter_dtypes != {torch.float32}:
        raise RuntimeError(
            f"FP32 diagnostic model has unexpected dtypes: {parameter_dtypes}")

    print(f"checkpoint={args.checkpoint}")
    print(f"checkpoint_floating_dtypes={checkpoint_dtypes}")
    print(f"completed_epoch={completed_epoch}")
    print("simple_gate_blends=" + ",".join(
        f"{blend:.4f}" for blend in blends))
    print(f"parameter_dtype={next(model.parameters()).dtype}")
    print(f"device={device}")

    # Importing verification loads MXNet and the .bin decoding dependencies, so
    # keep it out of module import to allow lightweight unit testing.
    from eval import verification

    targets = args.targets or list(cfg.val_targets)
    failures = []
    for target in targets:
        dataset_path = os.path.join(cfg.rec, target + ".bin")
        if not os.path.exists(dataset_path):
            failures.append(f"{target}: missing {dataset_path}")
            print(f"[FAIL] {failures[-1]}")
            continue
        print(f"\nTesting {target} in FP32 from {dataset_path}")
        dataset = verification.load_bin(dataset_path, (112, 112))
        try:
            _, _, accuracy, std, xnorm, _ = verification.test(
                dataset,
                model,
                args.batch_size,
                args.nfolds,
                fail_on_nonfinite=True,
            )
        except FloatingPointError as error:
            failures.append(f"{target}: {error}")
            print(f"[FAIL] {failures[-1]}")
            continue
        print(
            f"[PASS] {target}: accuracy={accuracy:.5f}+-{std:.5f}, "
            f"xnorm={xnorm:.6f}"
        )

    if failures:
        print("\nFP32 checkpoint test failed:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("\nAll requested FP32 verification targets are finite.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

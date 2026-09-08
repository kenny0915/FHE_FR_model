"""Evaluate finite-gated face-verification canaries for a controlled model."""

from __future__ import annotations

import argparse
import os

import torch

from controlled_degree2.mine_deployment_tails import atomic_json_dump
from controlled_degree2.model import load_controlled_checkpoint
from controlled_degree2.train import evaluate_canaries, load_canaries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--canary-root", required=True)
    parser.add_argument("--sets", default="lfw")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    names = [name for name in args.sets.split(",") if name]
    if not names or args.batch_size <= 0:
        raise ValueError("at least one canary and a positive batch size are required")

    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    model, payload = load_controlled_checkpoint(args.checkpoint, device=device)
    canaries = load_canaries(args.canary_root, names)
    scores = evaluate_canaries(model, canaries, args.batch_size)
    if scores is None:
        raise FloatingPointError("controlled checkpoint failed finite canary gate")
    result = {
        "format": "controlled-degree2-canary-v1",
        "checkpoint": os.path.abspath(args.checkpoint),
        "checkpoint_degree": int(payload["degree"]),
        "sets": scores,
        "all_finite": True,
    }
    atomic_json_dump(result, args.output)
    print(f"saved {os.path.abspath(args.output)}: {scores}", flush=True)


if __name__ == "__main__":
    main()

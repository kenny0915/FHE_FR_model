#!/usr/bin/env python3
"""CLI for no-training polynomial stability simulation."""

import argparse
import importlib.util
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from backbones import get_model
from stability_analysis.workflow import AnalysisConfig, analyze, replace_activations


BASELINES = {
    "ms1mv3_r50": "r50",
    "ms1mv3_poolformer_s24": "poolformer_s24",
}


def _extract_backbone_state(checkpoint):
    if not isinstance(checkpoint, dict):
        raise TypeError("checkpoint must contain a state dictionary")
    for key in ("state_dict_backbone", "state_dict", "model"):
        if isinstance(checkpoint.get(key), dict):
            checkpoint = checkpoint[key]
            break
    if checkpoint and all(key.startswith("module.") for key in checkpoint):
        checkpoint = {key[7:]: value for key, value in checkpoint.items()}
    return checkpoint


def _factory_from_file(path):
    spec = importlib.util.spec_from_file_location("user_polynomial", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "make_activation"):
        raise AttributeError("activation file must define make_activation(name, original_module)")
    return module.make_activation


def _loader(dataset, batch_size, max_samples):
    if dataset == "synthetic":
        tensor = torch.linspace(-1, 1, max_samples * 3 * 112 * 112).reshape(max_samples, 3, 112, 112)
        return DataLoader(tensor, batch_size=batch_size)
    root = Path(dataset)
    if (root / "train.rec").exists() and (root / "train.idx").exists():
        from dataset import MXFaceDataset
        data = MXFaceDataset(str(root), local_rank=0)
    else:
        from torchvision import transforms
        from torchvision.datasets import ImageFolder
        data = ImageFolder(root, transforms.Compose([
            transforms.Resize((112, 112)), transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ]))
    data = torch.utils.data.Subset(data, range(min(max_samples, len(data))))
    return DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=0)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=BASELINES, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True, help="ImageFolder, MS1M RecordIO directory, or synthetic")
    parser.add_argument("--activation-file", help="Python file defining make_activation(name, original_module)")
    parser.add_argument("--input-scale", type=float, default=6.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-batches", type=int, default=10)
    parser.add_argument("--max-samples", type=int, default=80)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no-backward", action="store_true")
    parser.add_argument("--output", default="stability_report.json")
    return parser


def main():
    args = build_parser().parse_args()
    baseline = get_model(BASELINES[args.model], fp16=False, num_features=512)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    baseline.load_state_dict(_extract_backbone_state(checkpoint), strict=True)
    factory = _factory_from_file(args.activation_file) if args.activation_file else None
    polynomial, replacements = replace_activations(
        baseline, factory=factory, input_scale=args.input_scale)
    report = analyze(
        baseline, polynomial,
        _loader(args.dataset, args.batch_size, args.max_samples),
        replacements, device=args.device,
        config=AnalysisConfig(
            max_batches=args.max_batches,
            interval=(-args.input_scale, args.input_scale),
            backward=not args.no_backward,
        ),
    )
    report["run"] = vars(args)
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"], indent=2))
    print(f"report={args.output}")


if __name__ == "__main__":
    main()

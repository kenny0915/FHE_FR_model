#!/usr/bin/env python3
"""Run isolated layer replacement and approximation-interval studies."""

import argparse
import importlib.util
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from backbones import get_model
from stability_analysis.activations import (
    make_herpn_for,
    make_uniform_quadratic_for,
)
from stability_analysis.layerwise import analyze_layerwise
from stability_analysis.reporting import write_outputs


MODELS = {
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
        raise AttributeError(
            "activation file must define make_activation(name, original_module)")
    return module.make_activation


def _synthetic_data(sample_count, seed):
    generator = torch.Generator().manual_seed(seed)
    random = torch.rand(
        sample_count, 3, 112, 112, generator=generator) * 2.0 - 1.0
    axis = torch.linspace(-1.0, 1.0, 112)
    horizontal = axis.reshape(1, 1, 1, 112)
    vertical = axis.reshape(1, 1, 112, 1)
    pattern = 0.15 * torch.sin(4.0 * horizontal) * torch.cos(3.0 * vertical)
    return (0.85 * random + pattern).clamp(-1.0, 1.0)


def _loader(dataset, batch_size, max_samples, seed):
    if dataset == "synthetic":
        return DataLoader(
            _synthetic_data(max_samples, seed), batch_size=batch_size,
            shuffle=False), "synthetic"

    root = Path(dataset)
    if (root / "train.rec").exists() and (root / "train.idx").exists():
        from dataset import MXFaceDataset
        data = MXFaceDataset(str(root), local_rank=0)
        dataset_kind = "recordio"
    else:
        from torchvision import transforms
        from torchvision.datasets import ImageFolder
        data = ImageFolder(root, transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ]))
        dataset_kind = "imagefolder"
    data = torch.utils.data.Subset(data, range(min(max_samples, len(data))))
    return DataLoader(
        data, batch_size=batch_size, shuffle=False, num_workers=0), dataset_kind


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--dataset", required=True,
        help="ImageFolder, RecordIO directory, or synthetic")
    parser.add_argument(
        "--replacement", choices=("herpn", "uniform-quadratic"),
        default="herpn")
    parser.add_argument(
        "--activation-file",
        help="Python file defining make_activation(name, original_module)")
    parser.add_argument("--input-scale", type=float, default=6.0)
    parser.add_argument(
        "--interval-sweep", type=float, nargs="+",
        default=(0.5, 1, 2, 4, 6))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-batches", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=32)
    parser.add_argument("--max-activation-samples", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--output-prefix", default="stability_analysis/results/study")
    return parser


def _factory(args):
    if args.activation_file:
        return _factory_from_file(args.activation_file)
    if args.replacement == "uniform-quadratic":
        return lambda _name, module: make_uniform_quadratic_for(
            module, args.input_scale)
    return lambda _name, module: make_herpn_for(module, args.input_scale)


def main():
    args = build_parser().parse_args()
    if args.input_scale <= 0:
        raise ValueError("--input-scale must be positive")
    if any(scale <= 0 for scale in args.interval_sweep):
        raise ValueError("all --interval-sweep values must be positive")

    model = get_model(MODELS[args.model], fp16=False, num_features=512)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(_extract_backbone_state(checkpoint), strict=True)
    loader, dataset_kind = _loader(
        args.dataset, args.batch_size, args.max_samples, args.seed)

    def progress(name, current, total):
        print("[{}/{}] {}".format(current, total, name), flush=True)

    report = analyze_layerwise(
        model,
        loader,
        device=args.device,
        max_batches=args.max_batches,
        interval=(-args.input_scale, args.input_scale),
        interval_scales=tuple(sorted(set(args.interval_sweep))),
        max_samples=args.max_activation_samples,
        factory=_factory(args),
        progress=progress,
    )
    report["run"] = vars(args)
    report["run"]["dataset_kind"] = dataset_kind
    prefix = Path(args.output_prefix)
    write_outputs(
        report,
        str(prefix) + ".json",
        markdown_path=str(prefix) + ".md",
        csv_path=str(prefix) + ".csv",
    )
    print("json={}.json".format(prefix))
    print("markdown={}.md".format(prefix))
    print("csv={}.csv".format(prefix))


if __name__ == "__main__":
    main()

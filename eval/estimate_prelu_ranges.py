# coding: utf-8

import argparse
import csv
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from skimage import transform as trans

from backbones import get_model, iresnet50


class PReLURangeEstimator:
    def __init__(self, model, sample_limit=200000):
        self.sample_limit = int(sample_limit)
        self.stats = {}
        self.handles = []
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.PReLU):
                self.stats[name] = {
                    "count": 0,
                    "min": float("inf"),
                    "max": float("-inf"),
                    "absmax": 0.0,
                    "samples": [],
                }
                self.handles.append(module.register_forward_hook(self._make_hook(name)))

    def _make_hook(self, name):
        def hook(module, inputs, output):
            if not inputs or not torch.is_tensor(inputs[0]):
                return
            with torch.no_grad():
                x = inputs[0].detach().float()
                finite = x[torch.isfinite(x)]
                if finite.numel() == 0:
                    return
                item = self.stats[name]
                item["count"] += int(finite.numel())
                item["min"] = min(item["min"], float(finite.min().item()))
                item["max"] = max(item["max"], float(finite.max().item()))
                item["absmax"] = max(item["absmax"], float(finite.abs().max().item()))
                if self.sample_limit > 0:
                    flat = finite.reshape(-1)
                    take = min(flat.numel(), self.sample_limit)
                    if flat.numel() > take:
                        step = max(1, flat.numel() // take)
                        flat = flat[::step][:take]
                    item["samples"].append(flat.cpu())
        return hook

    def close(self):
        for handle in self.handles:
            handle.remove()

    def rows(self, margin):
        rows = []
        for name, item in self.stats.items():
            if item["count"] == 0:
                continue
            samples = torch.cat(item["samples"]) if item["samples"] else torch.empty(0)
            if samples.numel():
                q = torch.quantile(
                    samples.abs(),
                    torch.tensor([0.99, 0.999, 0.9999], dtype=samples.dtype),
                )
                p99, p999, p9999 = [float(v) for v in q]
            else:
                p99 = p999 = p9999 = float("nan")
            rows.append({
                "layer": name,
                "count": item["count"],
                "min": item["min"],
                "max": item["max"],
                "absmax": item["absmax"],
                "abs_p99": p99,
                "abs_p999": p999,
                "abs_p9999": p9999,
                "scale_p999_margin": margin * p999,
                "scale_absmax_margin": margin * item["absmax"],
            })
        return rows


def load_model(args, device):
    network = "r50" if args.network in ("iresnet50", "r50") else args.network
    if network == "r50":
        model = iresnet50(False, dropout=0, fp16=False)
    else:
        model = get_model(network, dropout=0, fp16=False)
    if args.model_prefix:
        state = torch.load(args.model_prefix, map_location="cpu")
        model.load_state_dict(state)
    return model.to(device).eval()


def align_image(img, landmark):
    src = np.array([
        [30.2946, 51.6963],
        [65.5318, 51.5014],
        [48.0252, 71.7366],
        [33.5493, 92.3655],
        [62.7299, 92.2041],
    ], dtype=np.float32)
    src[:, 0] += 8.0
    tform = trans.SimilarityTransform()
    tform.estimate(landmark, src)
    warped = cv2.warpAffine(img, tform.params[0:2, :], (112, 112), borderValue=0.0)
    warped = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
    flipped = np.fliplr(warped)
    batch = np.stack([warped, flipped], axis=0).transpose(0, 3, 1, 2)
    return batch.astype(np.float32) / 255.0 / 0.5 - 1.0


def iter_ijb_batches(image_path, target, batch_size, max_images):
    list_path = os.path.join(image_path, "meta", "{}_name_5pts_score.txt".format(target.lower()))
    crop_dir = os.path.join(image_path, "loose_crop")
    with open(list_path) as f:
        lines = f.readlines()
    if max_images > 0:
        lines = lines[:max_images]
    batch = []
    for line in lines:
        parts = line.strip().split(" ")
        img = cv2.imread(os.path.join(crop_dir, parts[0]))
        if img is None:
            continue
        landmark = np.array([float(x) for x in parts[1:-1]], dtype=np.float32).reshape(5, 2)
        pair = align_image(img, landmark)
        batch.extend([pair[0], pair[1]])
        if len(batch) >= batch_size:
            yield torch.from_numpy(np.stack(batch[:batch_size], axis=0))
            batch = batch[batch_size:]
    if batch:
        yield torch.from_numpy(np.stack(batch, axis=0))


def iter_random_batches(batch_size, batches):
    for _ in range(batches):
        yield torch.empty(batch_size, 3, 112, 112).uniform_(-1.0, 1.0)


def write_rows(rows, output_csv):
    fieldnames = [
        "layer", "count", "min", "max", "absmax",
        "abs_p99", "abs_p999", "abs_p9999",
        "scale_p999_margin", "scale_absmax_margin",
    ]
    if output_csv:
        os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    writer = csv.DictWriter(os.sys.stdout, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Estimate original PReLU input ranges for ResNet-style models.")
    parser.add_argument("--model-prefix", default="work_dirs/ms1mv3_r50/model.pt")
    parser.add_argument("--network", default="r50")
    parser.add_argument("--image-path", default="", help="IJB root containing meta/ and loose_crop/; uses random inputs when empty")
    parser.add_argument("--target", default="IJBC", choices=["IJBB", "IJBC"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-images", type=int, default=512)
    parser.add_argument("--random-batches", type=int, default=4)
    parser.add_argument("--sample-limit", type=int, default=200000)
    parser.add_argument("--margin", type=float, default=1.1)
    parser.add_argument("--output-csv", default="work_dirs/prelu_input_ranges.csv")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    model = load_model(args, device)
    estimator = PReLURangeEstimator(model, sample_limit=args.sample_limit)

    if args.image_path:
        batches = iter_ijb_batches(args.image_path, args.target, args.batch_size, args.max_images)
    else:
        batches = iter_random_batches(args.batch_size, args.random_batches)

    with torch.no_grad():
        for batch_index, batch in enumerate(batches):
            print("batch {} shape {}".format(batch_index, tuple(batch.shape)), flush=True)
            model(batch.to(device))

    rows = estimator.rows(args.margin)
    estimator.close()
    write_rows(rows, args.output_csv)
    if args.output_csv:
        print("Wrote {}".format(args.output_csv))


if __name__ == "__main__":
    main()

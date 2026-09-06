"""Fit direct quadratics on the *same per-channel ranges* used by run10.

This is intentionally not a fresh range search.  ``--reference-run10`` is a
required control input: its deployed ``lam_fit`` and ``lam_reg`` buffers are
copied exactly, while a new histogram-weighted degree-2 fit is computed from
the confirmed ResNet-50 teacher.  Consequently, polynomial degree is the only
activation-definition variable between the reference and this experiment.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from typing import Dict

import numpy as np
import torch

from controlled_degree2.model import checkpoint_state, load_teacher, prelu_names


class ChannelAbsHistogram:
    """Per-channel histogram of absolute activation inputs on logarithmic bins."""

    def __init__(
        self,
        channels: int,
        hist_min: float,
        hist_max: float,
        bins: int,
        device: torch.device,
    ):
        self.channels = int(channels)
        self.hist_min = float(hist_min)
        self.hist_max = float(hist_max)
        self.log_min = math.log10(self.hist_min)
        self.log_max = math.log10(self.hist_max)
        self.bins = int(bins)
        self.counts = torch.zeros(
            channels, bins, dtype=torch.float64, device=device
        )
        self.maximum = torch.zeros(channels, dtype=torch.float32, device=device)
        self.total = 0
        self._channel_offsets = (
            torch.arange(channels, device=device, dtype=torch.long) * bins
        )[:, None]

    @torch.no_grad()
    def update(self, inputs: torch.Tensor) -> None:
        absolute = inputs.detach().float().abs().transpose(0, 1)
        absolute = absolute.reshape(self.channels, -1)
        scaled = (
            absolute.clamp_min(self.hist_min).log10() - self.log_min
        ) / (self.log_max - self.log_min)
        indices = (scaled * self.bins).long().clamp_(0, self.bins - 1)
        flat = (indices + self._channel_offsets).reshape(-1)
        self.counts += torch.bincount(
            flat, minlength=self.channels * self.bins
        ).reshape(self.channels, self.bins).double()
        self.maximum = torch.maximum(self.maximum, absolute.amax(dim=1))
        self.total += absolute.shape[1]

    def bin_geometry(self):
        device = self.counts.device
        edges = 10 ** (
            self.log_min
            + torch.arange(self.bins + 1, dtype=torch.float64, device=device)
            / self.bins
            * (self.log_max - self.log_min)
        )
        return (edges[:-1] * edges[1:]).sqrt(), edges[1:] - edges[:-1]


def weighted_quadratic_abs_fit(
    probabilities: torch.Tensor,
    centers: torch.Tensor,
    widths: torch.Tensor,
    lam_fit: torch.Tensor,
    fit_eps: float = 0.05,
    slope: torch.Tensor | None = None,
):
    """Fit ``abs(x) ~= d0+d2*x^2`` and return x-unit coeffs and error.

    The distribution is assumed sign-symmetric, matching run10's calibration.
    Error is the expected relative activation error after training-time clipping
    at ``lam_fit``; deployment remains unclipped.
    """
    probabilities = probabilities.double()
    centers = centers.double()
    widths = widths.double()
    lam_fit = lam_fit.double().reshape(-1)
    lower = centers / (centers[1] / centers[0]).sqrt()
    normalized = centers[None, :] / lam_fit[:, None]
    inside = (lower[None, :] <= lam_fit[:, None]).double()
    weights = (
        probabilities + fit_eps * widths[None, :] / lam_fit[:, None]
    ) * inside
    u2 = normalized.square()
    m0 = weights.sum(dim=1)
    m2 = (weights * u2).sum(dim=1)
    m4 = (weights * u2.square()).sum(dim=1)
    matrix = torch.stack(
        (torch.stack((m0, m2), dim=-1), torch.stack((m2, m4), dim=-1)),
        dim=-2,
    )
    ridge = 1e-8
    matrix = matrix + ridge * torch.eye(
        2, dtype=matrix.dtype, device=matrix.device
    ) * (m0[:, None, None] + ridge)
    rhs = torch.stack(
        ((weights * normalized).sum(dim=1), (weights * normalized**3).sum(dim=1)),
        dim=-1,
    )
    unit_coefficients = torch.linalg.solve(matrix, rhs)

    approximation = (
        unit_coefficients[:, :1]
        + unit_coefficients[:, 1:] * normalized.square()
    )
    if slope is None:
        slope = torch.full_like(lam_fit, -1.0)
    else:
        slope = slope.to(device=lam_fit.device, dtype=torch.float64).reshape(-1)
    even_weight = (1.0 - slope) / 2.0
    signed_energy = (1.0 + slope.square()) / 2.0
    in_error = even_weight[:, None] * lam_fit[:, None] * (
        normalized - approximation
    )
    out_error = (centers[None, :] - lam_fit[:, None]).clamp_min(0)
    numerator = (
        probabilities
        * (
            inside * in_error.square()
            + (1.0 - inside) * signed_energy[:, None] * out_error.square()
        )
    ).sum(dim=1)
    denominator = (
        probabilities * centers[None, :].square() * signed_energy[:, None]
    ).sum(dim=1).clamp_min(1e-30)
    relative_error = (numerator / denominator).sqrt()

    coefficients = torch.stack(
        (unit_coefficients[:, 0] * lam_fit, unit_coefficients[:, 1] / lam_fit),
        dim=1,
    )
    return coefficients, relative_error


def _torch_load(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        return torch.load(path, map_location="cpu", weights_only=False)


def reference_ranges(path: str, names, channels: Dict[str, int]):
    """Read the actual deployed range buffers, not potentially stale metadata."""
    payload = _torch_load(path)
    state = checkpoint_state(payload)
    calibration = payload.get("poly_calib", {}) if isinstance(payload, dict) else {}
    ranges = {}
    for name in names:
        fit_key, reg_key = f"{name}.lam_fit", f"{name}.lam_reg"
        if fit_key in state and reg_key in state:
            lam_fit = state[fit_key].detach().float().reshape(-1)
            lam_reg = state[reg_key].detach().float().reshape(-1)
        elif name in calibration:
            lam_fit = torch.as_tensor(calibration[name]["lam_fit"]).float().reshape(-1)
            lam_reg = torch.as_tensor(calibration[name]["lam_reg"]).float().reshape(-1)
        else:
            raise KeyError(f"reference checkpoint has no ranges for {name}")
        if lam_fit.numel() != channels[name] or lam_reg.numel() != channels[name]:
            raise ValueError(
                f"reference range shape for {name} does not match teacher channels"
            )
        if not torch.isfinite(lam_fit).all() or not bool((lam_fit > 0).all()):
            raise ValueError(f"reference lam_fit for {name} is invalid")
        scale = float(calibration.get(name, {}).get("lam_scale", 1.0))
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError(f"reference lam_scale for {name} is invalid")
        ranges[name] = (lam_fit, lam_reg, scale)
    return ranges, payload.get("format") if isinstance(payload, dict) else None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", required=True)
    parser.add_argument(
        "--reference-run10",
        required=True,
        help="run10 student_best.pt; read-only source of exact per-channel ranges",
    )
    parser.add_argument("--dataset-root", required=True, help="directory with train.rec/train.idx")
    parser.add_argument("--out", required=True)
    parser.add_argument("--num-images", type=int, default=100000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--hist-min", type=float, default=1e-3)
    parser.add_argument("--hist-max", type=float, default=1e4)
    parser.add_argument("--bins", type=int, default=8192)
    parser.add_argument("--fit-eps", type=float, default=0.05)
    parser.add_argument(
        "--swap-rgb",
        action="store_true",
        help="only for a nonstandard RecordIO reader returning BGR; repo dataset.py returns RGB",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.num_images <= 0:
        raise SystemExit("--num-images must be positive")
    device = torch.device(args.device)
    teacher = load_teacher(args.teacher, device=device).eval()
    names = prelu_names(teacher)
    channels = {
        name: int(teacher.get_submodule(name).weight.numel()) for name in names
    }
    ranges, reference_format = reference_ranges(
        args.reference_run10, names, channels
    )

    histograms = {
        name: ChannelAbsHistogram(
            channels[name], args.hist_min, args.hist_max, args.bins, device
        )
        for name in names
    }
    memory_mib = sum(hist.counts.numel() for hist in histograms.values()) * 8 / 2**20
    print(
        f"{len(names)} activations, {sum(channels.values())} channels, "
        f"histogram memory {memory_mib:.0f} MiB"
    )
    handles = []
    for name in names:
        handles.append(
            teacher.get_submodule(name).register_forward_hook(
                lambda _module, inputs, _output, name=name: histograms[name].update(inputs[0])
            )
        )

    # Import lazily so pure calibration tests do not require MXNet.
    from dataset import MXFaceDataset

    dataset = MXFaceDataset(args.dataset_root, local_rank=0)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=args.num_workers > 0,
    )
    seen, started = 0, time.time()
    with torch.no_grad():
        for images, _labels in loader:
            if args.swap_rgb:
                images = images.flip(1)
            teacher(images.to(device, non_blocking=True))
            seen += images.shape[0]
            if seen % (20 * args.batch_size) < args.batch_size:
                print(f"  {seen}/{args.num_images} images ({time.time() - started:.0f}s)")
            if seen >= args.num_images:
                break
    for handle in handles:
        handle.remove()

    layers = {}
    print("\nactivation          C  lam_min  lam_med  lam_max  fit_err  worst")
    for name in names:
        histogram = histograms[name]
        lam_fit, lam_reg, lam_scale = ranges[name]
        lam_fit = lam_fit.to(device=device, dtype=torch.float64)
        # run10 fitted at its initial calibrated interval and subsequently
        # widened q by q_s(x)=s*q(x/s).  Reproduce that order, rather than
        # silently giving degree 2 the advantage of a fresh fit at the final
        # interval.
        fit_lam = lam_fit / lam_scale
        probabilities = histogram.counts / histogram.counts.sum(
            dim=1, keepdim=True
        ).clamp_min(1)
        centers, widths = histogram.bin_geometry()
        even, relative_error = weighted_quadratic_abs_fit(
            probabilities,
            centers,
            widths,
            fit_lam,
            args.fit_eps,
            teacher.get_submodule(name).weight.detach(),
        )
        even[:, 0] *= lam_scale
        even[:, 1] /= lam_scale
        maximum = histogram.maximum.double()
        ratio = maximum / lam_fit
        entry = {
            "channels": channels[name],
            "lam_fit": lam_fit.cpu().tolist(),
            "lam_reg": lam_reg.double().cpu().tolist(),
            "even_coeffs": even.cpu().tolist(),
            "fit_relerr": relative_error.cpu().tolist(),
            "fit_relerr_median": float(relative_error.median()),
            "fit_relerr_max": float(relative_error.max()),
            "lam_min": float(lam_fit.min()),
            "lam_median": float(lam_fit.median()),
            "lam_max": float(lam_fit.max()),
            "teacher_max": maximum.cpu().tolist(),
            "max_ratio": float(ratio.max()),
            "lam_scale": lam_scale,
            "fit_interval_before_scale": fit_lam.cpu().tolist(),
            "reference_range_exact": True,
        }
        layers[name] = entry
        print(
            f"{name:<18}{channels[name]:>5} {entry['lam_min']:>8.3f}"
            f" {entry['lam_median']:>8.3f} {entry['lam_max']:>8.3f}"
            f" {entry['fit_relerr_median']:>8.3f} {entry['fit_relerr_max']:>7.3f}"
        )

    output = {
        "meta": {
            "experiment": "controlled direct degree-2 vs run10 degree-4",
            "degree": 2,
            "polynomial": "c0+c1*x+c2*x^2",
            "approximation_target": "per-channel PReLU",
            "interval": "run10 initial fit interval followed by the same widening; exact deployed lam buffers",
            "reference_run10": os.path.abspath(args.reference_run10),
            "reference_format": reference_format,
            "teacher": os.path.abspath(args.teacher),
            "num_images_seen": seen,
            "fit_eps": args.fit_eps,
            "hist_min": args.hist_min,
            "hist_max": args.hist_max,
            "bins": args.bins,
            "input_order": "bgr" if args.swap_rgb else "rgb",
            "multiplicative_depth_per_activation": 1,
        },
        "layers": layers,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

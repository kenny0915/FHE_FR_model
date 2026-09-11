"""Compare every per-channel quadratic activation in two checkpoints.

The controlled degree-2 models store one polynomial per activation channel::

    q_c(x) = c0_c + c1_c * x + c2_c * x^2

This tool reports the actual symmetric approximation interval
``[-lam_fit_c, lam_fit_c]`` and plots both the raw-x curve and the curve
normalized to each channel's own interval.  It accepts the native controlled
checkpoint format, the older Nano4 ``state_dict`` wrapper, and plain state
dictionaries exported for evaluation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch


STATE_KEYS = ("state_dict_backbone", "state_dict", "model", "model_state_dict")
METADATA_EXCLUDE = {"state_dict_backbone", "state_dict", "model", "model_state_dict", "canary"}


@dataclass(frozen=True)
class ActivationData:
    name: str
    coeffs: np.ndarray
    lam_fit: np.ndarray
    lam_reg: np.ndarray
    slope: np.ndarray

    @property
    def channels(self) -> int:
        return int(self.coeffs.shape[0])


@dataclass(frozen=True)
class CheckpointData:
    path: Path
    label: str
    metadata: dict[str, Any]
    state: Mapping[str, torch.Tensor]
    activations: dict[str, ActivationData]


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.item()
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        if len(value) > 64:
            return {"length": len(value), "preview": [_jsonable(v) for v in value[:4]]}
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def extract_state_dict(payload: Any) -> Mapping[str, torch.Tensor]:
    """Extract a tensor state dictionary from supported checkpoint wrappers."""
    if not isinstance(payload, Mapping):
        raise TypeError(f"checkpoint must contain a mapping, got {type(payload)!r}")
    for key in STATE_KEYS:
        candidate = payload.get(key)
        if isinstance(candidate, Mapping) and candidate:
            if all(isinstance(value, torch.Tensor) for value in candidate.values()):
                return candidate
    if payload and all(isinstance(value, torch.Tensor) for value in payload.values()):
        return payload
    raise ValueError("could not find a tensor state dictionary in checkpoint")


def degree2_coefficients(tensor: torch.Tensor, name: str) -> np.ndarray:
    """Return ``[C, 3]`` coefficients, accepting a zero-padded cubic slot."""
    coeffs = tensor.detach().float().cpu()
    if coeffs.ndim != 2 or coeffs.shape[1] not in (3, 4):
        raise ValueError(f"{name} must have shape [C, 3] or [C, 4], got {tuple(coeffs.shape)}")
    if coeffs.shape[1] == 4:
        cubic_absmax = float(coeffs[:, 3].abs().max())
        if cubic_absmax > 1e-8:
            raise ValueError(
                f"{name} is not degree 2: padded cubic coefficient absmax={cubic_absmax:g}"
            )
        coeffs = coeffs[:, :3]
    return coeffs.numpy()


def _natural_key(name: str) -> list[Any]:
    return [int(piece) if piece.isdigit() else piece for piece in re.split(r"(\d+)", name)]


def _activation_site_key(name: str) -> tuple[Any, ...]:
    """Order the stem first, followed by residual stages and block indices."""
    if name == "prelu":
        return (0, 0)
    match = re.fullmatch(r"layer(\d+)\.(\d+)\.prelu", name)
    if match:
        return (int(match.group(1)), int(match.group(2)) + 1)
    return (999, *_natural_key(name))


def activation_data(state: Mapping[str, torch.Tensor]) -> dict[str, ActivationData]:
    names = sorted(
        (key[: -len(".coeffs")] for key in state if key.endswith(".coeffs")),
        key=_activation_site_key,
    )
    if not names:
        raise ValueError("state dictionary has no '*.coeffs' polynomial activations")
    result: dict[str, ActivationData] = {}
    for name in names:
        coeffs = degree2_coefficients(state[f"{name}.coeffs"], f"{name}.coeffs")
        channels = coeffs.shape[0]

        def vector(suffix: str) -> np.ndarray:
            key = f"{name}.{suffix}"
            if key not in state:
                raise KeyError(f"missing required activation buffer {key!r}")
            value = state[key].detach().float().cpu().reshape(-1).numpy()
            if value.size != channels:
                raise ValueError(f"{key} has {value.size} values; expected {channels}")
            return value

        lam_fit = vector("lam_fit")
        lam_reg = vector("lam_reg")
        slope = vector("slope")
        if not np.all(np.isfinite(coeffs)):
            raise ValueError(f"{name}.coeffs contains a non-finite value")
        if not np.all(np.isfinite(lam_fit)) or np.any(lam_fit <= 0):
            raise ValueError(f"{name}.lam_fit must be finite and positive")
        result[name] = ActivationData(name, coeffs, lam_fit, lam_reg, slope)
    return result


def load_checkpoint(path: Path, label: str) -> CheckpointData:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = extract_state_dict(payload)
    metadata: dict[str, Any] = {}
    if isinstance(payload, Mapping) and state is not payload:
        metadata = {
            str(key): _jsonable(value)
            for key, value in payload.items()
            if key not in METADATA_EXCLUDE
        }
    activations = activation_data(state)
    if isinstance(payload, Mapping):
        metadata["poly_calib_buffer_consistency"] = calibration_buffer_consistency(
            payload.get("poly_calib"), activations
        )
    return CheckpointData(path, label, metadata, state, activations)


def calibration_buffer_consistency(
    poly_calib: Any, activations: Mapping[str, ActivationData]
) -> dict[str, Any]:
    """Check whether descriptive calibration metadata matches deployed buffers."""
    result: dict[str, Any] = {}
    if not isinstance(poly_calib, Mapping):
        return {"available": False}
    for field in ("lam_fit", "lam_reg"):
        mismatched_sites: list[str] = []
        maximum_difference = 0.0
        comparable_sites = 0
        for name, data in activations.items():
            item = poly_calib.get(name)
            if not isinstance(item, Mapping) or field not in item:
                continue
            metadata_values = np.asarray(item[field], dtype=np.float64).reshape(-1)
            buffer_values = np.asarray(getattr(data, field), dtype=np.float64).reshape(-1)
            if metadata_values.shape != buffer_values.shape:
                mismatched_sites.append(name)
                continue
            comparable_sites += 1
            difference = float(np.max(np.abs(metadata_values - buffer_values)))
            maximum_difference = max(maximum_difference, difference)
            if not np.allclose(metadata_values, buffer_values, rtol=1e-6, atol=1e-7):
                mismatched_sites.append(name)
        result[field] = {
            "comparable_sites": comparable_sites,
            "mismatched_sites": mismatched_sites,
            "maximum_absolute_difference": maximum_difference,
        }
    result["available"] = True
    return result


def evaluate(coeffs: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Evaluate channelwise quadratics with broadcastable ``x``."""
    return coeffs[:, 0, None] + coeffs[:, 1, None] * x + coeffs[:, 2, None] * x * x


def quantiles(values: np.ndarray) -> dict[str, float]:
    points = np.quantile(values, (0.0, 0.1, 0.5, 0.9, 1.0))
    return dict(zip(("min", "p10", "median", "p90", "max"), map(float, points)))


def _safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan, dtype=np.float64),
        where=np.abs(denominator) > 1e-15,
    )


def compare_activation(left: ActivationData, right: ActivationData) -> dict[str, Any]:
    if left.channels != right.channels:
        raise ValueError(
            f"channel mismatch at {left.name}: {left.channels} versus {right.channels}"
        )
    u = np.linspace(-1.0, 1.0, 401, dtype=np.float64)[None, :]
    y_left = evaluate(left.coeffs, u * left.lam_fit[:, None]) / left.lam_fit[:, None]
    y_right = evaluate(right.coeffs, u * right.lam_fit[:, None]) / right.lam_fit[:, None]
    normalized_rmse = np.sqrt(np.mean(np.square(y_left - y_right), axis=1))
    overlap = np.minimum(left.lam_fit, right.lam_fit)
    x_overlap = u * overlap[:, None]
    raw_overlap_rmse = np.sqrt(
        np.mean(np.square(evaluate(left.coeffs, x_overlap) - evaluate(right.coeffs, x_overlap)), axis=1)
    )
    return {
        "site": left.name,
        "channels": left.channels,
        "left_lam_fit": quantiles(left.lam_fit),
        "right_lam_fit": quantiles(right.lam_fit),
        "right_over_left_lam_fit": quantiles(right.lam_fit / left.lam_fit),
        "left_lam_reg_over_fit": quantiles(left.lam_reg / left.lam_fit),
        "right_lam_reg_over_fit": quantiles(right.lam_reg / right.lam_fit),
        "left_c0": quantiles(left.coeffs[:, 0]),
        "right_c0": quantiles(right.coeffs[:, 0]),
        "left_c1": quantiles(left.coeffs[:, 1]),
        "right_c1": quantiles(right.coeffs[:, 1]),
        "left_c2": quantiles(left.coeffs[:, 2]),
        "right_c2": quantiles(right.coeffs[:, 2]),
        "right_over_left_c0": quantiles(_safe_ratio(right.coeffs[:, 0], left.coeffs[:, 0])),
        "right_over_left_c2": quantiles(_safe_ratio(right.coeffs[:, 2], left.coeffs[:, 2])),
        "c1_abs_difference": quantiles(np.abs(right.coeffs[:, 1] - left.coeffs[:, 1])),
        "normalized_curve_rmse": quantiles(normalized_rmse),
        "raw_curve_rmse_on_overlap": quantiles(raw_overlap_rmse),
    }


def compare_checkpoints(left: CheckpointData, right: CheckpointData) -> list[dict[str, Any]]:
    left_names = set(left.activations)
    right_names = set(right.activations)
    if left_names != right_names:
        raise ValueError(
            "activation sites differ: "
            f"only left={sorted(left_names - right_names)}, "
            f"only right={sorted(right_names - left_names)}"
        )
    names = sorted(left_names, key=_activation_site_key)
    return [compare_activation(left.activations[name], right.activations[name]) for name in names]


def state_layout(state: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    coeff_shapes = sorted({tuple(value.shape) for key, value in state.items() if key.endswith(".coeffs")})
    return {
        "entries": len(state),
        "tensor_values": int(sum(value.numel() for value in state.values())),
        "running_mean_entries": sum(key.endswith("running_mean") for key in state),
        "running_var_entries": sum(key.endswith("running_var") for key in state),
        "num_batches_tracked_entries": sum(key.endswith("num_batches_tracked") for key in state),
        "coefficient_shapes": [list(shape) for shape in coeff_shapes],
    }


def _flatten_row(row: dict[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, Mapping):
            for subkey, subvalue in value.items():
                flat[f"{key}_{subkey}"] = subvalue
        else:
            flat[key] = value
    return flat


def write_csv(rows: Sequence[dict[str, Any]], path: Path) -> None:
    flat = [_flatten_row(row) for row in rows]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)


def _representative_channel(left: ActivationData, right: ActivationData) -> int:
    ratio = right.lam_fit / left.lam_fit
    score = (
        np.abs(np.log(left.lam_fit / np.median(left.lam_fit)))
        + np.abs(np.log(right.lam_fit / np.median(right.lam_fit)))
        + np.abs(np.log(ratio / np.median(ratio)))
    )
    return int(np.argmin(score))


def _normalized_curves(data: ActivationData, u: np.ndarray) -> np.ndarray:
    x = data.lam_fit[:, None] * u[None, :]
    return evaluate(data.coeffs, x) / data.lam_fit[:, None]


def _plot_band(axis, u: np.ndarray, curves: np.ndarray, color: str, label: str) -> None:
    p10, median, p90 = np.quantile(curves, (0.1, 0.5, 0.9), axis=0)
    axis.fill_between(u, p10, p90, color=color, alpha=0.18, linewidth=0)
    axis.plot(u, median, color=color, linewidth=1.5, label=label)


def plot_overview(left: CheckpointData, right: CheckpointData, output: Path) -> None:
    import matplotlib.pyplot as plt

    names = sorted(left.activations, key=_activation_site_key)
    u = np.linspace(-1.0, 1.0, 401)
    columns = 5
    rows = math.ceil(len(names) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(18, 15), sharex=True, sharey=True)
    axes = np.asarray(axes).reshape(-1)
    colors = ("#2676B8", "#D35400")
    for index, name in enumerate(names):
        axis = axes[index]
        left_data, right_data = left.activations[name], right.activations[name]
        _plot_band(axis, u, _normalized_curves(left_data, u), colors[0], left.label)
        _plot_band(axis, u, _normalized_curves(right_data, u), colors[1], right.label)
        target = np.where(u >= 0, u, np.median(left_data.slope) * u)
        axis.plot(u, target, color="black", linestyle="--", linewidth=0.9, alpha=0.65)
        axis.axvline(0, color="#999999", linewidth=0.4)
        axis.axhline(0, color="#999999", linewidth=0.4)
        ratio = np.median(right_data.lam_fit / left_data.lam_fit)
        axis.set_title(f"{name}\nC={left_data.channels}, median interval ratio={ratio:.3f}", fontsize=9)
        axis.grid(alpha=0.15)
    for axis in axes[len(names) :]:
        axis.axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.962),
        ncol=2,
        frameon=False,
    )
    figure.suptitle(
        "All 25 degree-2 activation sites (median and channel P10–P90)\n"
        "Normalized coordinates: u=x/lam_fit, v=q(x)/lam_fit; dashed=PReLU target",
        y=0.995,
        fontsize=14,
    )
    figure.supxlabel("normalized input u")
    figure.supylabel("normalized output v")
    figure.tight_layout(rect=(0.02, 0.03, 1, 0.925))
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_intervals(left: CheckpointData, right: CheckpointData, output: Path) -> None:
    import matplotlib.pyplot as plt

    names = sorted(left.activations, key=_activation_site_key)
    x = np.arange(len(names), dtype=float)
    colors = ("#2676B8", "#D35400")
    figure, axis = plt.subplots(figsize=(17, 7.5))
    for offset, checkpoint, color in (
        (-0.12, left, colors[0]),
        (+0.12, right, colors[1]),
    ):
        fit_q = np.array(
            [np.quantile(checkpoint.activations[name].lam_fit, (0.1, 0.5, 0.9)) for name in names]
        )
        reg_median = np.array(
            [np.median(checkpoint.activations[name].lam_reg) for name in names]
        )
        axis.errorbar(
            x + offset,
            fit_q[:, 1],
            yerr=np.stack((fit_q[:, 1] - fit_q[:, 0], fit_q[:, 2] - fit_q[:, 1])),
            fmt="o",
            markersize=5,
            capsize=2,
            linewidth=1.2,
            color=color,
            label=f"{checkpoint.label}: lam_fit median [P10, P90]",
        )
        axis.scatter(
            x + offset,
            reg_median,
            marker="x",
            s=26,
            color=color,
            alpha=0.8,
            label=f"{checkpoint.label}: lam_reg median",
        )
    axis.set_yscale("log")
    axis.set_xticks(x, names, rotation=65, ha="right")
    axis.set_ylabel("interval half-width (activation x units, log scale)")
    axis.set_title(
        "Per-layer symmetric intervals: approximation [-lam_fit, lam_fit] "
        "and training control [-lam_reg, lam_reg]"
    )
    axis.grid(axis="y", which="both", alpha=0.2)
    axis.legend(ncol=2, frameon=False, fontsize=9)
    figure.tight_layout()
    figure.savefig(output, dpi=190)
    plt.close(figure)


def plot_detailed_pdf(left: CheckpointData, right: CheckpointData, output: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    colors = ("#2676B8", "#D35400")
    u = np.linspace(-1.0, 1.0, 401)
    names = sorted(left.activations, key=_activation_site_key)
    with PdfPages(output) as pdf:
        for page, name in enumerate(names, start=1):
            left_data, right_data = left.activations[name], right.activations[name]
            channel = _representative_channel(left_data, right_data)
            figure = plt.figure(figsize=(11.7, 8.3))
            grid = figure.add_gridspec(2, 2, height_ratios=(1.15, 1.0))
            raw_axis = figure.add_subplot(grid[0, :])
            normalized_axis = figure.add_subplot(grid[1, 0])
            interval_axis = figure.add_subplot(grid[1, 1])

            max_lam = max(left_data.lam_fit[channel], right_data.lam_fit[channel])
            x_target = np.linspace(-max_lam, max_lam, 501)
            target = np.where(x_target >= 0, x_target, left_data.slope[channel] * x_target)
            raw_axis.plot(x_target, target, "k--", linewidth=1.1, label="PReLU target")
            equations = []
            for data, color in ((left_data, colors[0]), (right_data, colors[1])):
                lam = float(data.lam_fit[channel])
                x = np.linspace(-lam, lam, 501)
                coeffs = data.coeffs[channel]
                y = coeffs[0] + coeffs[1] * x + coeffs[2] * x * x
                raw_axis.plot(x, y, color=color, linewidth=2, label=data is left_data and left.label or right.label)
                raw_axis.axvline(-lam, color=color, linestyle=":", linewidth=0.9)
                raw_axis.axvline(lam, color=color, linestyle=":", linewidth=0.9)
                equations.append(
                    f"{data is left_data and left.label or right.label}: "
                    f"q(x)={coeffs[0]:.6g} + {coeffs[1]:.6g}x + {coeffs[2]:.6g}x², "
                    f"lam_fit={lam:.6g}, lam_reg={data.lam_reg[channel]:.6g}"
                )
            raw_axis.set_title(f"Representative paired channel {channel}: raw-x polynomial")
            raw_axis.set_xlabel("activation input x")
            raw_axis.set_ylabel("q(x)")
            raw_axis.grid(alpha=0.2)
            raw_axis.legend(frameon=False)
            raw_axis.text(
                0.01,
                0.98,
                "\n".join(equations),
                transform=raw_axis.transAxes,
                va="top",
                fontsize=8,
                bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
            )

            _plot_band(
                normalized_axis,
                u,
                _normalized_curves(left_data, u),
                colors[0],
                left.label,
            )
            _plot_band(
                normalized_axis,
                u,
                _normalized_curves(right_data, u),
                colors[1],
                right.label,
            )
            normalized_target = np.where(u >= 0, u, np.median(left_data.slope) * u)
            normalized_axis.plot(u, normalized_target, "k--", linewidth=1, label="PReLU target")
            normalized_axis.set_title("All channels in own normalized interval")
            normalized_axis.set_xlabel("u=x/lam_fit")
            normalized_axis.set_ylabel("q(x)/lam_fit")
            normalized_axis.grid(alpha=0.2)
            normalized_axis.legend(frameon=False, fontsize=8)

            order = np.argsort(left_data.lam_fit)
            ranks = np.arange(left_data.channels)
            interval_axis.plot(
                ranks,
                left_data.lam_fit[order],
                color=colors[0],
                linewidth=1.2,
                label=f"{left.label} lam_fit",
            )
            interval_axis.plot(
                ranks,
                right_data.lam_fit[order],
                color=colors[1],
                linewidth=1.2,
                label=f"{right.label} lam_fit",
            )
            interval_axis.plot(
                ranks,
                left_data.lam_reg[order],
                color=colors[0],
                linestyle=":",
                linewidth=0.9,
                label=f"{left.label} lam_reg",
            )
            interval_axis.plot(
                ranks,
                right_data.lam_reg[order],
                color=colors[1],
                linestyle=":",
                linewidth=0.9,
                label=f"{right.label} lam_reg",
            )
            interval_axis.set_yscale("log")
            interval_axis.set_title("Every channel (ordered by left lam_fit)")
            interval_axis.set_xlabel("channel rank")
            interval_axis.set_ylabel("interval half-width")
            interval_axis.grid(alpha=0.2, which="both")
            interval_axis.legend(frameon=False, fontsize=7)

            ratio = right_data.lam_fit / left_data.lam_fit
            figure.suptitle(
                f"{name} — page {page}/{len(names)} — C={left_data.channels}\n"
                f"median lam_fit: {left.label}={np.median(left_data.lam_fit):.5g}, "
                f"{right.label}={np.median(right_data.lam_fit):.5g}, "
                f"right/left={np.median(ratio):.4f}",
                fontsize=13,
            )
            figure.tight_layout(rect=(0, 0, 1, 0.93))
            pdf.savefig(figure, dpi=160)
            plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--left-label", default="left")
    parser.add_argument("--right-label", default="right")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    left = load_checkpoint(args.left, args.left_label)
    right = load_checkpoint(args.right, args.right_label)
    rows = compare_checkpoints(left, right)
    write_csv(rows, args.output_dir / "layer_summary.csv")
    summary = {
        "polynomial": "q_c(x) = c0_c + c1_c*x + c2_c*x^2",
        "approximation_interval": "per-channel symmetric [-lam_fit_c, lam_fit_c]",
        "left": {
            "label": left.label,
            "path": str(left.path),
            "metadata": left.metadata,
            "state_layout": state_layout(left.state),
        },
        "right": {
            "label": right.label,
            "path": str(right.path),
            "metadata": right.metadata,
            "state_layout": state_layout(right.state),
        },
        "layers": rows,
    }
    with (args.output_dir / "comparison_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    plot_overview(left, right, args.output_dir / "quadratic_shapes_by_layer.png")
    plot_intervals(left, right, args.output_dir / "approximation_intervals_by_layer.png")
    plot_detailed_pdf(left, right, args.output_dir / "quadratic_comparison_by_layer.pdf")
    print(f"Compared {len(rows)} activation sites; outputs written to {args.output_dir}")


if __name__ == "__main__":
    main()

# coding: utf-8

import csv
import os

import matplotlib
import numpy as np
import torch

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from eval.non_linear_replacement import PReLU_Approx, PreciseReLUAlpha10, THORPolynomialGELU


def collect_prelu_slopes(module):
    slopes = []
    for child in module.modules():
        if isinstance(child, torch.nn.PReLU):
            slopes.append(child.weight.detach().float().cpu().reshape(-1))
    if not slopes:
        return torch.tensor([0.25], dtype=torch.float32)
    return torch.cat(slopes)


def write_activation_comparison_plot(model, save_dir, input_scale=1.0):
    os.makedirs(save_dir, exist_ok=True)
    slopes = collect_prelu_slopes(model)
    slope_mean = float(slopes.mean())
    slope_min = float(slopes.min())
    slope_max = float(slopes.max())
    input_scale = float(input_scale)
    x = torch.linspace(-2.0 * input_scale, 2.0 * input_scale, 4001)
    precise = PreciseReLUAlpha10(input_scale=input_scale).eval()(x).detach().cpu().numpy()
    x_np = x.numpy()
    relu = np.maximum(x_np, 0.0)
    prelu_mean = np.where(x_np >= 0.0, x_np, slope_mean * x_np)
    prelu_min = np.where(x_np >= 0.0, x_np, slope_min * x_np)
    prelu_max = np.where(x_np >= 0.0, x_np, slope_max * x_np)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, xlim, title in (
            (axes[0], (-input_scale, input_scale), "Scaled approximation range"),
            (axes[1], (-2.0 * input_scale, 2.0 * input_scale), "Outside scaled range behavior")):
        ax.plot(x_np, precise, label="PreciseReLU alpha=10", linewidth=2.0)
        ax.plot(x_np, relu, label="ReLU", linestyle="--", linewidth=1.4)
        ax.plot(x_np, prelu_mean, label="trained PReLU mean slope", linewidth=1.4)
        ax.fill_between(x_np, prelu_min, prelu_max, alpha=0.18, label="trained PReLU min/max slope")
        ax.axvline(-input_scale, color="gray", linestyle=":", linewidth=1.0)
        ax.axvline(input_scale, color="gray", linestyle=":", linewidth=1.0)
        ax.set_xlim(*xlim)
        ax.set_ylim(-0.75 * input_scale, 2.25 * input_scale)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
        ax.set_title(title)
        ax.set_xlabel("activation input")
        ax.set_ylabel("activation output")
    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle("IResNet50 activation replacement: scaled PreciseReLU alpha=10 vs original trained PReLU")
    fig.tight_layout()
    out_path = os.path.join(save_dir, "precise_relu_alpha10_vs_prelu.png")
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print("Saved activation comparison plot to {}".format(out_path))
    print("PReLU slope stats: min={:.6g}, mean={:.6g}, max={:.6g}, count={}".format(
        slope_min, slope_mean, slope_max, slopes.numel()))


class ActivationDebugRecorder:
    _next_id = 0

    def __init__(self, model, debug_dir, max_batches):
        self.debug_dir = debug_dir
        self.max_batches = int(max_batches)
        self.batch_index = 0
        self.handles = []
        self.saved_nonfinite = set()
        self.recorder_id = ActivationDebugRecorder._next_id
        ActivationDebugRecorder._next_id += 1
        os.makedirs(debug_dir, exist_ok=True)
        self.csv_path = os.path.join(debug_dir, "activation_stats.csv")
        write_header = not os.path.exists(self.csv_path)
        self.csv_file = open(self.csv_path, "a", newline="")
        self.fieldnames = [
            "recorder_id", "batch", "module", "kind", "shape", "numel",
            "finite", "nan", "posinf", "neginf", "finite_min", "finite_max",
            "finite_mean", "finite_std", "finite_absmax",
        ]
        self.writer = csv.DictWriter(self.csv_file, fieldnames=self.fieldnames)
        if write_header:
            self.writer.writeheader()
        for name, module in model.named_modules():
            if isinstance(module, (PReLU_Approx, THORPolynomialGELU)):
                self.handles.append(module.register_forward_hook(self._make_hook(name)))
        print("Recording activation stats for {} modules into {}".format(
            len(self.handles), self.csv_path))

    def _enabled(self):
        return self.max_batches <= 0 or self.batch_index < self.max_batches

    def _make_hook(self, name):
        def hook(module, inputs, output):
            if not self._enabled() or not inputs:
                return
            self._write_stats(name, "input", inputs[0])
            self._write_stats(name, "output", output)
        return hook

    def _write_stats(self, name, kind, tensor):
        if not torch.is_tensor(tensor):
            return
        with torch.no_grad():
            t = tensor.detach()
            finite_mask = torch.isfinite(t)
            finite_count = int(finite_mask.sum().item())
            nan_count = int(torch.isnan(t).sum().item())
            posinf_count = int(torch.isposinf(t).sum().item())
            neginf_count = int(torch.isneginf(t).sum().item())
            numel = int(t.numel())
            if finite_count:
                finite = t[finite_mask].float()
                finite_min = float(finite.min().item())
                finite_max = float(finite.max().item())
                finite_mean = float(finite.mean().item())
                finite_std = float(finite.std(unbiased=False).item())
                finite_absmax = float(finite.abs().max().item())
            else:
                finite_min = finite_max = finite_mean = finite_std = finite_absmax = float("nan")
            self.writer.writerow({
                "recorder_id": self.recorder_id,
                "batch": self.batch_index,
                "module": name,
                "kind": kind,
                "shape": "x".join(str(v) for v in t.shape),
                "numel": numel,
                "finite": finite_count,
                "nan": nan_count,
                "posinf": posinf_count,
                "neginf": neginf_count,
                "finite_min": finite_min,
                "finite_max": finite_max,
                "finite_mean": finite_mean,
                "finite_std": finite_std,
                "finite_absmax": finite_absmax,
            })
            if finite_count != numel:
                print("Non-finite {} detected at batch {}, module {}: nan={}, +inf={}, -inf={}".format(
                    kind, self.batch_index, name, nan_count, posinf_count, neginf_count))
                key = (name, kind)
                if key not in self.saved_nonfinite:
                    self.saved_nonfinite.add(key)
                    if t.ndim > 0:
                        sample = t[:min(8, t.shape[0])].detach().float().cpu().numpy()
                    else:
                        sample = t.detach().float().cpu().numpy()
                    safe_name = name.replace(".", "_").replace("/", "_")
                    out_path = os.path.join(
                        self.debug_dir,
                        "nonfinite_{}_{}_batch{}.npz".format(safe_name, kind, self.batch_index),
                    )
                    np.savez_compressed(out_path, values=sample)
                    print("Saved first non-finite {} sample to {}".format(kind, out_path))
            self.csv_file.flush()

    def write_model_output_stats(self, tensor):
        if self._enabled():
            self._write_stats("model_output", "output", tensor)

    def step(self):
        self.batch_index += 1

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.csv_file.close()

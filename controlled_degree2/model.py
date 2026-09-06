"""Direct per-channel quadratic replacement used by the controlled experiment.

The target is the teacher's PReLU on each channel and on that channel's
calibrated symmetric interval ``[-lam_fit, lam_fit]``::

    PReLU_a(x) = (1 + a) / 2 * x + (1 - a) / 2 * abs(x)
    q(x)        = c0 + c1 * x + c2 * x^2

Only ``abs(x)`` is approximated.  The linear component is exact.  Deployment
does not clamp or branch and the quadratic costs one multiplicative level.
Training may clamp at ``lam_fit`` and adds a differentiable range penalty, as
in the run10 recipe; those aids are disabled automatically in eval mode.
"""

from __future__ import annotations

import copy
import json
import os
import re
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


FORMAT = "fhe-fr/controlled-direct-degree2-v1"


def as_channel_tensor(value, channels: int) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32).reshape(-1)
    if tensor.numel() == 1:
        tensor = tensor.expand(channels).clone()
    if tensor.numel() != channels:
        raise ValueError(
            f"expected one value or {channels} channel values, got {tensor.numel()}"
        )
    if not torch.isfinite(tensor).all() or not bool((tensor > 0).all()):
        raise ValueError("activation intervals must be finite and positive")
    return tensor


def uniform_even_abs_coefficients(lam, channels: int) -> torch.Tensor:
    """Return ``[d0, d2]`` for the uniform L2 fit to ``abs(x)``.

    The exact normal-equation solution on ``[-1, 1]`` is
    ``abs(u) ~= 3/16 + 15/16*u^2``.  Scaling by ``lam`` gives the fit in x
    units.  Calibration normally replaces this with histogram-weighted values.
    """
    lam = as_channel_tensor(lam, channels)
    return torch.stack((3.0 * lam / 16.0, 15.0 / (16.0 * lam)), dim=1)


def prelu_to_quadratic_coefficients(
    slope: torch.Tensor,
    lam,
    even_coeffs=None,
) -> torch.Tensor:
    """Return per-channel ``[c0, c1, c2]`` approximating a PReLU."""
    slope = slope.detach().float().reshape(-1)
    channels = slope.numel()
    if even_coeffs is None:
        even = uniform_even_abs_coefficients(lam, channels)
    else:
        even = torch.as_tensor(even_coeffs, dtype=torch.float32).reshape(channels, 2)
    linear = (1.0 + slope) / 2.0
    even_weight = (1.0 - slope) / 2.0
    return torch.stack(
        (even_weight * even[:, 0], linear, even_weight * even[:, 1]), dim=1
    )


class DirectQuadratic(nn.Module):
    """Per-channel ``c0 + c1*x + c2*x^2`` with run10 training guard rails."""

    degree = 2

    def __init__(
        self,
        channels: int,
        lam_fit=1.0,
        lam_reg=None,
        coeffs: Optional[torch.Tensor] = None,
        slope=0.25,
        name: str = "",
    ):
        super().__init__()
        lam_reg = lam_fit if lam_reg is None else lam_reg
        slope_tensor = torch.as_tensor(slope, dtype=torch.float32).reshape(-1)
        if slope_tensor.numel() == 1:
            slope_tensor = slope_tensor.expand(channels).clone()
        if coeffs is None:
            coeffs = prelu_to_quadratic_coefficients(slope_tensor, lam_fit)
        coeffs = torch.as_tensor(coeffs, dtype=torch.float32)
        if coeffs.shape != (channels, 3):
            raise ValueError(f"coeffs must have shape {(channels, 3)}, got {coeffs.shape}")
        if not torch.isfinite(coeffs).all():
            raise ValueError("quadratic coefficients must be finite")

        self.coeffs = nn.Parameter(coeffs.clone(), requires_grad=False)
        self.register_buffer("lam_fit", as_channel_tensor(lam_fit, channels))
        self.register_buffer("lam_reg", as_channel_tensor(lam_reg, channels))
        self.register_buffer("slope", slope_tensor.float().reshape(channels))
        self.name = name

        self.alpha = 1.0
        self.clip = True
        self.clip_eval = False
        self.penalty = "hinge"
        self.gamma = 10.0
        self.last_penalty = None
        self.last_oor = 0.0
        self.last_max = 0.0

    @property
    def channels(self) -> int:
        return int(self.coeffs.shape[0])

    def _view(self, x: torch.Tensor):
        return (1, self.channels) + (1,) * (x.ndim - 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        raw = x.float()
        work = raw
        lam_fit = self.lam_fit.reshape(self._view(work))

        if self.training:
            lam_reg = self.lam_reg.reshape(self._view(work))
            detached_abs = raw.detach().abs()
            self.last_max = float((detached_abs / lam_fit).amax())
            self.last_oor = float((detached_abs > lam_reg).float().mean())
            if self.penalty == "pillar":
                self.last_penalty = ((raw / lam_reg) ** self.gamma).mean()
            elif self.penalty == "hinge":
                excess = F.relu(raw.abs() / lam_reg - 1.0)
                self.last_penalty = excess.square().sum() / raw.shape[0]
            else:
                raise ValueError(f"unknown range penalty {self.penalty!r}")
            if self.clip:
                work = torch.maximum(torch.minimum(work, lam_fit), -lam_fit)
        elif self.clip_eval:
            work = torch.maximum(torch.minimum(work, lam_fit), -lam_fit)

        coefficients = self.coeffs.t().reshape((3,) + self._view(work))
        c0, c1, c2 = coefficients
        output = c0 + c1 * work + c2 * (work * work)
        if self.alpha < 1.0 and self.training:
            output = self.alpha * output + (1.0 - self.alpha) * F.prelu(raw, self.slope)
        return output.to(input_dtype)

    def extra_repr(self) -> str:
        return (
            f"channels={self.channels}, lam_fit=[{float(self.lam_fit.min()):.3g}.."
            f"{float(self.lam_fit.max()):.3g}], name={self.name!r}"
        )


def prelu_names(model: nn.Module) -> List[str]:
    return [name for name, module in model.named_modules() if isinstance(module, nn.PReLU)]


def quadratic_modules(model: nn.Module) -> Iterable[DirectQuadratic]:
    return (module for module in model.modules() if isinstance(module, DirectQuadratic))


def _set_module(root: nn.Module, dotted_name: str, replacement: nn.Module) -> None:
    parent_name, _, leaf = dotted_name.rpartition(".")
    parent = root.get_submodule(parent_name) if parent_name else root
    setattr(parent, leaf, replacement)


def replace_prelu_with_quadratic(
    model: nn.Module, calibration: Mapping[str, Mapping]
) -> List[str]:
    names = prelu_names(model)
    if not names:
        raise ValueError("model has no PReLU activations to replace")
    missing = [name for name in names if name not in calibration]
    if missing:
        raise KeyError(f"missing calibration for {len(missing)} activations: {missing[:5]}")
    for name in names:
        prelu = model.get_submodule(name)
        entry = calibration[name]
        coefficients = prelu_to_quadratic_coefficients(
            prelu.weight, entry["lam_fit"], entry.get("even_coeffs")
        )
        _set_module(
            model,
            name,
            DirectQuadratic(
                channels=coefficients.shape[0],
                lam_fit=entry["lam_fit"],
                lam_reg=entry["lam_reg"],
                coeffs=coefficients,
                slope=prelu.weight.detach(),
                name=name,
            ),
        )
    return names


def set_quadratic_schedule(
    model: nn.Module,
    *,
    alpha=None,
    gamma: Optional[float] = None,
    clip: Optional[bool] = None,
    clip_eval: Optional[bool] = None,
    penalty: Optional[str] = None,
) -> None:
    for module in quadratic_modules(model):
        if alpha is not None:
            module.alpha = float(alpha[module.name] if isinstance(alpha, dict) else alpha)
        if gamma is not None:
            module.gamma = float(gamma)
        if clip is not None:
            module.clip = bool(clip)
        if clip_eval is not None:
            module.clip_eval = bool(clip_eval)
        if penalty is not None:
            module.penalty = str(penalty)


def set_lam_reg_ratio(model: nn.Module, ratio: float) -> None:
    if not 0.0 < ratio <= 1.0:
        raise ValueError("lam_reg_ratio must be in (0, 1]")
    with torch.no_grad():
        for module in quadratic_modules(model):
            module.lam_reg.copy_(module.lam_fit * ratio)


def collect_range_stats(model: nn.Module):
    penalties, oor, maxima = [], {}, {}
    for module in quadratic_modules(model):
        if module.last_penalty is not None:
            penalties.append(module.last_penalty)
            oor[module.name] = module.last_oor
            maxima[module.name] = module.last_max
    if not penalties:
        raise RuntimeError("no quadratic range statistics were collected")
    return torch.stack(penalties).mean(), oor, maxima


def parse_lam_scale(spec: str) -> Dict[str, float]:
    result = {}
    for item in re.split(r"[+;,\s]+", spec or ""):
        if item:
            key, value = item.split(":", 1)
            result[key.strip()] = float(value)
    return result


def _matching_scale(name: str, scales: Mapping[str, float]):
    matches = [
        (len(key), value)
        for key, value in scales.items()
        if key == "all" or key == name or name.startswith(key + ".")
    ]
    return max(matches)[1] if matches else None


@torch.no_grad()
def scale_intervals(
    model: nn.Module,
    calibration: Optional[MutableMapping[str, MutableMapping]],
    scales: Mapping[str, float],
) -> Dict[str, float]:
    """Rescale q to ``s*q(x/s)`` while widening both interval bounds by s."""
    touched = {}
    for module in quadratic_modules(model):
        scale = _matching_scale(module.name, scales)
        if scale is None or scale == 1.0:
            continue
        if scale <= 0 or not np.isfinite(scale):
            raise ValueError("interval scale factors must be finite and positive")
        module.lam_fit.mul_(scale)
        module.lam_reg.mul_(scale)
        module.coeffs[:, 0].mul_(scale)
        module.coeffs[:, 2].div_(scale)
        if calibration is not None and module.name in calibration:
            entry = calibration[module.name]
            for key in ("lam_fit", "lam_reg"):
                entry[key] = (np.asarray(entry[key], dtype=np.float64) * scale).tolist()
            if "even_coeffs" in entry:
                even = np.asarray(entry["even_coeffs"], dtype=np.float64)
                even[:, 0] *= scale
                even[:, 1] /= scale
                entry["even_coeffs"] = even.tolist()
            entry["lam_scale"] = float(entry.get("lam_scale", 1.0) * scale)
        touched[module.name] = scale
    return touched


def load_calibration(path: str) -> dict:
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    layers = payload.get("layers", payload)
    degree = payload.get("meta", {}).get("degree", 2)
    if degree != 2:
        raise ValueError(f"expected degree-2 calibration, got degree={degree}")
    return layers


def build_controlled_iresnet50(**kwargs) -> nn.Module:
    """Build the eval architecture; state loading supplies calibrated buffers."""
    from backbones.iresnet import iresnet50

    model = iresnet50(pretrained=False, **kwargs)
    placeholder = {}
    for name in prelu_names(model):
        channels = model.get_submodule(name).weight.numel()
        placeholder[name] = {"lam_fit": [1.0] * channels, "lam_reg": [1.0] * channels}
    replace_prelu_with_quadratic(model, placeholder)
    return model


def checkpoint_state(payload) -> dict:
    if not isinstance(payload, dict):
        raise TypeError(f"checkpoint must be a dict, got {type(payload)!r}")
    for key in ("state_dict_backbone", "state_dict", "model"):
        if isinstance(payload.get(key), dict):
            payload = payload[key]
            break
    if payload and all(name.startswith("module.") for name in payload):
        payload = {name[len("module.") :]: value for name, value in payload.items()}
    return payload


def load_teacher(weights: str, device="cpu") -> nn.Module:
    """Load the current repo's baseline iResNet-50 teacher strictly."""
    from backbones.iresnet import iresnet50

    try:
        payload = torch.load(weights, map_location="cpu", weights_only=True)
    except Exception:
        payload = torch.load(weights, map_location="cpu", weights_only=False)
    model = iresnet50(pretrained=False, dropout=0, fp16=False)
    model.load_state_dict(checkpoint_state(payload), strict=True)
    return model.to(device)


def save_checkpoint(
    path: str,
    model: nn.Module,
    calibration: dict,
    *,
    teacher_weights: str,
    extra: Optional[dict] = None,
) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    bare_model = model.module if hasattr(model, "module") else model
    state = {name: value.detach().cpu() for name, value in bare_model.state_dict().items()}
    payload = {
        "format": FORMAT,
        "degree": 2,
        "polynomial": "c0+c1*x+c2*x^2",
        "approximation_target": "per-channel PReLU",
        "interval": "per-channel symmetric [-lam_fit, lam_fit]",
        "multiplicative_depth_per_activation": 1,
        "state_dict_backbone": state,
        "poly_calib": copy.deepcopy(calibration),
        "teacher_weights": os.path.abspath(teacher_weights),
        "network": "r50_controlled_d2",
        "input_order": "rgb",
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_controlled_checkpoint(path: str, device="cpu"):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != FORMAT or payload.get("degree") != 2:
        raise ValueError(f"{path} is not a controlled direct degree-2 checkpoint")
    model = build_controlled_iresnet50(dropout=0, fp16=False)
    model.load_state_dict(checkpoint_state(payload), strict=True)
    return model.to(device), payload

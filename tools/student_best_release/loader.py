"""Self-contained loader for the folded degree-2 student checkpoint."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Dict, Tuple

import torch
from torch import nn


CHECKPOINT_FORMAT = "adaface-nano4/poly-v1"


class Degree2Polynomial(nn.Module):
    """Channel-wise c0 + c1*x + c2*x^2 activation.

    The checkpoint retains a fourth, legacy c4 storage column. It must be zero
    for a degree-2 model and is deliberately not evaluated.
    """

    def __init__(self, channels: int, trainable: bool = False):
        super().__init__()
        self.coeffs = nn.Parameter(
            torch.zeros(channels, 4), requires_grad=trainable
        )
        self.register_buffer("lam_fit", torch.ones(channels))
        self.register_buffer("lam_reg", torch.ones(channels))
        self.register_buffer("slope", torch.zeros(channels))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        work = inputs.float()
        shape = (1, self.coeffs.shape[0]) + (1,) * (inputs.ndim - 2)
        c0, c1, c2, _legacy_c4 = self.coeffs.t().reshape((4,) + shape)
        output = c0 + c1 * work + c2 * work.square()
        return output.to(inputs.dtype)


class FoldedChannelAffine2d(nn.Module):
    """Frozen-statistics BatchNorm represented as channel-wise multiply/add."""

    def __init__(self, channels: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        shape = (1, -1) + (1,) * (inputs.ndim - 2)
        return inputs * self.weight.reshape(shape) + self.bias.reshape(shape)


def _import_backbone(path: Path):
    spec = importlib.util.spec_from_file_location("student_backbone", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import backbone from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _set_module(root: nn.Module, dotted_name: str, replacement: nn.Module) -> None:
    parent_name, _, leaf = dotted_name.rpartition(".")
    parent = root.get_submodule(parent_name) if parent_name else root
    setattr(parent, leaf, replacement)


def _add_convolution_bias(convolution: nn.Conv2d) -> None:
    if convolution.bias is None:
        convolution.bias = nn.Parameter(torch.zeros(convolution.out_channels))


def _build_folded_model(backbone, trainable_polynomials: bool) -> nn.Module:
    model = backbone.iresnet50(
        pretrained=False, dropout=0, fp16=False, arch_config="baseline"
    )
    activation_names = [
        name for name, module in model.named_modules()
        if isinstance(module, nn.PReLU)
    ]
    for name in activation_names:
        channels = model.get_submodule(name).weight.numel()
        _set_module(
            model, name,
            Degree2Polynomial(channels, trainable=trainable_polynomials),
        )

    _add_convolution_bias(model.conv1)
    model.bn1 = nn.Identity()
    for stage in (model.layer1, model.layer2, model.layer3, model.layer4):
        for block in stage:
            block.bn1 = FoldedChannelAffine2d(block.bn1.num_features)
            _add_convolution_bias(block.conv1)
            block.bn2 = nn.Identity()
            _add_convolution_bias(block.conv2)
            block.bn3 = nn.Identity()
            if block.downsample is not None:
                _add_convolution_bias(block.downsample[0])
                block.downsample[1] = nn.Identity()

    model.bn2 = FoldedChannelAffine2d(model.bn2.num_features)
    model.features = nn.Identity()
    return model


def load_model(
    checkpoint_path: str | Path = "model.pt",
    *,
    map_location: str | torch.device = "cpu",
    trainable_polynomials: bool = False,
) -> Tuple[nn.Module, Dict]:
    """Load the exact folded degree-2 inference graph and checkpoint metadata."""
    checkpoint_path = Path(checkpoint_path)
    try:
        payload = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
    except TypeError:
        payload = torch.load(checkpoint_path, map_location="cpu")

    if payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"unsupported checkpoint format: {payload.get('format')!r}")
    if payload.get("degree") != 2:
        raise ValueError(f"expected degree=2, got {payload.get('degree')!r}")
    if payload.get("bn_folded") is not True:
        raise ValueError("this loader requires a bn_folded=True checkpoint")

    state = payload.get("state_dict")
    if not isinstance(state, dict):
        raise ValueError("checkpoint has no state_dict")
    fourth_columns = [
        tensor[:, 3] for name, tensor in state.items()
        if name.endswith("coeffs")
    ]
    if not fourth_columns or any(torch.count_nonzero(column).item() for column in fourth_columns):
        raise ValueError("checkpoint contains a non-zero c4 term; it is not pure degree 2")

    backbone = _import_backbone(Path(__file__).with_name("backbone.py"))
    model = _build_folded_model(backbone, trainable_polynomials)
    model.load_state_dict(state, strict=True)
    model.to(map_location).eval()
    return model, payload


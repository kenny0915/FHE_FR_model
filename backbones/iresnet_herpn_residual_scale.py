"""From-scratch IResNet with HerPN activations and scaled residual branches.

For ``L`` residual blocks the default block is

    y = shortcut(x) + alpha * F(x),  alpha_0 = 1 / sqrt(L).

``alpha`` is a learned public scalar.  It adds no ciphertext-ciphertext
multiplication and can be folded into the final BatchNorm of ``F`` before FHE
inference.  HerPN is the AESPA degree-2 Hermite approximation of ReLU; after
BatchNorm calibration it folds exactly to ``A*x^2 + B*x + C``.

This backbone is intentionally trained from scratch.  It contains no PReLU
teacher, activation blend, or distillation path.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn

from .iresnet_no_relu import (
    FoldedHerPN,
    HerPN,
    IResNet as _BaseIResNet,
    conv1x1,
    conv3x3,
)

__all__ = [
    "PolynomialHerPN",
    "ResidualScaledIBasicBlock",
    "IResNet",
    "iresnet18",
    "iresnet34",
    "iresnet50",
    "iresnet100",
    "iresnet200",
]


class PolynomialHerPN(HerPN):
    """Pure HerPN activation with training-time interval diagnostics."""

    exclude_from_weight_decay = True
    is_polynomial_activation = True

    def __init__(self, channels, range_limit=6.0, eps=1e-4):
        if range_limit <= 0.0:
            raise ValueError("range_limit must be positive")
        super().__init__(channels, eps=eps)
        self.register_buffer(
            "range_limit", torch.tensor(float(range_limit), dtype=torch.float32))
        self._last_range_penalty = None
        self._last_input_absmax = None
        self._last_outside_fraction = None

    def forward(self, x):
        if self.training:
            compute_x = (
                x.float() if x.dtype in (torch.float16, torch.bfloat16) else x)
            limit = self.range_limit.to(device=x.device, dtype=compute_x.dtype)
            excess = F.relu(compute_x.abs() - limit)
            self._last_range_penalty = (
                excess.square().mean()
                + 0.1 * excess.flatten(1).amax(dim=1).square().mean()
            )
            self._last_input_absmax = compute_x.detach().abs().amax()
            self._last_outside_fraction = (
                (excess.detach() > 0.0).float().mean())
        else:
            self._last_range_penalty = None
            self._last_input_absmax = None
            self._last_outside_fraction = None
        return super().forward(x)

    def range_penalty(self):
        return self._last_range_penalty

    def range_stats(self):
        return {
            "absmax": self._last_input_absmax,
            "outside_fraction": self._last_outside_fraction,
        }

    @torch.no_grad()
    def folded(self):
        return FoldedHerPN.from_herpn(self)


class ResidualScaledIBasicBlock(nn.Module):
    """Pre-activation IResNet block with one public scale per residual path."""

    expansion = 1
    exclude_from_weight_decay = True

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 groups=1, base_width=64, dilation=1,
                 activation_factory=None, residual_scale_init=1.0,
                 residual_scale_trainable=True):
        super().__init__()
        if groups != 1 or base_width != 64:
            raise ValueError(
                "BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError(
                "Dilation > 1 not supported in BasicBlock")
        if not math.isfinite(float(residual_scale_init)):
            raise ValueError("residual_scale_init must be finite")
        if activation_factory is None:
            activation_factory = PolynomialHerPN

        self.bn1 = nn.BatchNorm2d(inplanes, eps=1e-5)
        self.conv1 = conv3x3(inplanes, planes)
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-5)
        # Preserve the historical name used by IResNet checkpoints and tools.
        self.prelu = activation_factory(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn3 = nn.BatchNorm2d(planes, eps=1e-5)
        self.downsample = downsample
        self.stride = stride
        self.residual_scale = nn.Parameter(
            torch.tensor(float(residual_scale_init), dtype=torch.float32),
            requires_grad=bool(residual_scale_trainable),
        )
        self.register_buffer(
            "residual_scale_initial",
            torch.tensor(float(residual_scale_init), dtype=torch.float32),
        )
        self._residual_scale_folded = False

    def forward_impl(self, x):
        identity = x
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        if self.residual_scale is not None:
            scale = self.residual_scale.to(device=out.device, dtype=out.dtype)
            out = scale * out
        if self.downsample is not None:
            identity = self.downsample(x)
        return identity + out

    def forward(self, x):
        # Checkpointing is deliberately omitted here.  The project-wide switch
        # is false by default, and a closure around a mutable scalar complicates
        # correct recomputation with no benefit for the intended R50 training.
        return self.forward_impl(x)

    @torch.no_grad()
    def fold_residual_scale_for_inference(self):
        """Fold alpha into bn3 affine parameters and remove its multiply."""
        if self.training or self.bn3.training:
            raise RuntimeError("Call eval() before folding residual scales")
        if self.residual_scale is None:
            return self
        scale = self.residual_scale.detach().to(
            device=self.bn3.weight.device, dtype=self.bn3.weight.dtype)
        self.bn3.weight.mul_(scale)
        self.bn3.bias.mul_(scale)
        self.register_parameter("residual_scale", None)
        self._residual_scale_folded = True
        return self


class IResNet(_BaseIResNet):
    """IResNet containing only HerPN nonlinearities and scaled residuals."""

    def __init__(self, block, layers, residual_scale_init=None,
                 residual_scale_trainable=True, **kwargs):
        if residual_scale_init is None:
            residual_scale_init = 1.0 / math.sqrt(sum(layers))
        object.__setattr__(
            self, "_residual_scale_init", float(residual_scale_init))
        object.__setattr__(
            self, "_residual_scale_trainable", bool(residual_scale_trainable))
        # The base class owns useful BN calibration and forward code.  Progress
        # is fixed at five because this model has no teacher branch to convert.
        kwargs["herpn_progress"] = 5.0
        super().__init__(block, layers, **kwargs)

    def _make_activation(self, channels, stage_name):
        del stage_name
        return PolynomialHerPN(
            channels=channels,
            range_limit=self.herpn_range_limit,
            eps=self.herpn_bn_eps,
        )

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False,
                    stage_name=None):
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion, eps=1e-5),
            )

        def make_block(block_stride, block_downsample, dilation):
            activation_factory = lambda channels: self._make_activation(
                channels, stage_name)
            return block(
                self.inplanes,
                planes,
                block_stride,
                block_downsample,
                self.groups,
                self.base_width,
                dilation,
                activation_factory=activation_factory,
                residual_scale_init=self._residual_scale_init,
                residual_scale_trainable=self._residual_scale_trainable,
            )

        layers = [make_block(stride, downsample, previous_dilation)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(make_block(1, None, self.dilation))
        return nn.Sequential(*layers)

    def set_herpn_progress(self, progress):
        if abs(float(progress) - 5.0) > 1e-6:
            raise ValueError(
                "Pure HerPN residual-scale models have fixed progress 5.0")
        self.herpn_progress.fill_(5.0)

    def polynomial_activations(self):
        return [
            module for module in self.modules()
            if isinstance(module, PolynomialHerPN)
        ]

    def residual_blocks(self):
        return [
            module for module in self.modules()
            if isinstance(module, ResidualScaledIBasicBlock)
        ]

    def herpn_range_penalty(self):
        penalties = [
            activation.range_penalty()
            for activation in self.polynomial_activations()
            if activation.range_penalty() is not None
        ]
        if not penalties:
            return next(self.parameters()).new_zeros(())
        return torch.stack(penalties).mean()

    def herpn_distillation_loss(self):
        # Kept for the trainer's shared metric interface.  There is no teacher.
        return next(self.parameters()).new_zeros(())

    def herpn_range_stats(self):
        return {
            name: module.range_stats()
            for name, module in self.named_modules()
            if isinstance(module, PolynomialHerPN)
        }

    def herpn_range_summary(self):
        stats = list(self.herpn_range_stats().values())
        absmax = [item["absmax"] for item in stats
                  if item["absmax"] is not None]
        outside = [item["outside_fraction"] for item in stats
                   if item["outside_fraction"] is not None]
        zero = next(self.parameters()).new_zeros(())
        return {
            "input_absmax": torch.stack(absmax).amax() if absmax else zero,
            "outside_fraction": (
                torch.stack(outside).mean() if outside else zero),
        }

    def residual_scale_summary(self):
        scales = [
            block.residual_scale.detach().float()
            for block in self.residual_blocks()
            if block.residual_scale is not None
        ]
        zero = next(self.parameters()).new_zeros(())
        if not scales:
            return {"min": zero, "max": zero, "mean": zero, "rms": zero}
        stacked = torch.stack(scales)
        return {
            "min": stacked.amin(),
            "max": stacked.amax(),
            "mean": stacked.mean(),
            "rms": stacked.square().mean().sqrt(),
        }

    @torch.no_grad()
    def fold_herpn_for_inference(self):
        """Fold every HerPN and residual scale to the FHE inference graph."""
        if self.training:
            raise RuntimeError("Call eval() before folding for inference")

        def replace_activations(module):
            for name, child in list(module.named_children()):
                if isinstance(child, PolynomialHerPN):
                    setattr(module, name, child.folded())
                else:
                    replace_activations(child)

        replace_activations(self)
        for block in self.residual_blocks():
            block.fold_residual_scale_for_inference()
        return self


def _iresnet(layers, pretrained, **kwargs):
    if pretrained:
        raise ValueError(
            "No pretrained pure-HerPN residual-scale checkpoint is bundled")
    return IResNet(ResidualScaledIBasicBlock, layers, **kwargs)


def iresnet18(pretrained=False, progress=True, **kwargs):
    del progress
    return _iresnet([2, 2, 2, 2], pretrained, **kwargs)


def iresnet34(pretrained=False, progress=True, **kwargs):
    del progress
    return _iresnet([3, 4, 6, 3], pretrained, **kwargs)


def iresnet50(pretrained=False, progress=True, **kwargs):
    del progress
    return _iresnet([3, 4, 14, 3], pretrained, **kwargs)


def iresnet100(pretrained=False, progress=True, **kwargs):
    del progress
    return _iresnet([3, 13, 30, 3], pretrained, **kwargs)


def iresnet200(pretrained=False, progress=True, **kwargs):
    del progress
    return _iresnet([6, 26, 60, 6], pretrained, **kwargs)

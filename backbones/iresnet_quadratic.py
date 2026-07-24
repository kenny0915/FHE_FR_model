"""IResNet with progressive PReLU-to-normalized-quadratic conversion.

The encrypted inference activation is a per-channel degree-2 polynomial

    z = x / S
    y = S * (a * z^2 + b * z + c)

where ``S`` defines the approximation interval ``[-S, S]``.  Division by S
is multiplication by a plaintext constant under FHE, so each activation uses
one sequential ciphertext-ciphertext multiplication for ``z^2``.

The coefficients are initialized from the pretrained channel-wise PReLU

    PReLU_s(x) = (1+s)/2*x + (1-s)/2*abs(x).

using the Gaussian-weighted degree-2 Hermite approximation

    abs(x) ~= k * (x^2 + 1),  k = 1/sqrt(2*pi).

This matches the approximately standardized inputs produced by the preceding
IResNet BatchNorm more closely than a uniform fit over the entire safety
interval.  All three coefficients remain independently learnable.
"""

import math

import torch
from torch import nn

from .iresnet_no_relu import IBasicBlock, IResNet as _ProgressiveIResNet

__all__ = [
    "NormalizedQuadratic",
    "FoldedNormalizedQuadratic",
    "ProgressiveQuadraticActivation",
    "IResNet",
    "iresnet18",
    "iresnet34",
    "iresnet50",
    "iresnet100",
    "iresnet200",
]

_STAGE_NAMES = ("stem", "layer1", "layer2", "layer3", "layer4")


def _prelu_initial_coefficients(
        slope, input_scale, abs_quadratic_coefficient):
    """Return normalized quadratic coefficients for a PReLU slope."""
    slope = slope.detach().clone().float().reshape(-1, 1, 1)
    linear = 0.5 * (1.0 + slope)
    even = 0.5 * (1.0 - slope)
    scale = float(input_scale)
    quadratic = scale * float(abs_quadratic_coefficient) * even
    constant = float(abs_quadratic_coefficient) * even / scale
    return quadratic, linear, constant


class NormalizedQuadratic(nn.Module):
    """Learnable channel-wise quadratic on a fixed normalized interval."""

    def __init__(self, channels, input_scale=6.0,
                 prelu_slope=0.25,
                 abs_quadratic_coefficient=1.0 / math.sqrt(2.0 * math.pi)):
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        if input_scale <= 0:
            raise ValueError("input_scale must be positive")
        slope = torch.full((channels,), float(prelu_slope))
        coefficient2, coefficient1, coefficient0 = (
            _prelu_initial_coefficients(
                slope, input_scale, abs_quadratic_coefficient))
        self.coefficient2 = nn.Parameter(coefficient2)
        self.coefficient1 = nn.Parameter(coefficient1)
        self.coefficient0 = nn.Parameter(coefficient0)
        self.register_buffer(
            "input_scale", torch.tensor(float(input_scale), dtype=torch.float32))

    @torch.no_grad()
    def initialize_from_prelu(
            self, slope,
            abs_quadratic_coefficient=1.0 / math.sqrt(2.0 * math.pi)):
        coefficient2, coefficient1, coefficient0 = (
            _prelu_initial_coefficients(
                slope, float(self.input_scale.item()),
                abs_quadratic_coefficient))
        self.coefficient2.copy_(coefficient2)
        self.coefficient1.copy_(coefficient1)
        self.coefficient0.copy_(coefficient0)

    def forward(self, x):
        compute_dtype = (
            torch.float32
            if x.dtype in (torch.float16, torch.bfloat16)
            else x.dtype
        )
        compute_x = x.to(dtype=compute_dtype)
        scale = self.input_scale.to(device=x.device, dtype=compute_dtype)
        # Multiplication by a public reciprocal is a plaintext operation in
        # the encrypted graph; no encrypted division is required.
        z = compute_x * scale.reciprocal()
        coefficient2 = self.coefficient2.to(dtype=compute_dtype)
        coefficient1 = self.coefficient1.to(dtype=compute_dtype)
        coefficient0 = self.coefficient0.to(dtype=compute_dtype)
        out = coefficient2 * z.square() + coefficient1 * z
        out = scale * (out + coefficient0)
        return out.to(dtype=x.dtype)

    @torch.no_grad()
    def folded_coefficients(self):
        """Return A, B, C for the exactly equivalent A*x^2+B*x+C."""
        scale = self.input_scale.to(
            device=self.coefficient2.device, dtype=self.coefficient2.dtype)
        return (
            self.coefficient2 / scale,
            self.coefficient1.detach().clone(),
            self.coefficient0 * scale,
        )


class FoldedNormalizedQuadratic(nn.Module):
    """Inference-only A*x^2+B*x+C representation."""

    def __init__(self, coefficient2, coefficient1, coefficient0):
        super().__init__()
        self.register_buffer("coefficient2", coefficient2.detach().clone())
        self.register_buffer("coefficient1", coefficient1.detach().clone())
        self.register_buffer("coefficient0", coefficient0.detach().clone())

    @classmethod
    def from_quadratic(cls, quadratic):
        return cls(*quadratic.folded_coefficients())

    def forward(self, x):
        compute_dtype = (
            torch.float32
            if x.dtype in (torch.float16, torch.bfloat16)
            else x.dtype
        )
        compute_x = x.to(dtype=compute_dtype)
        coefficient2 = self.coefficient2.to(dtype=compute_dtype)
        coefficient1 = self.coefficient1.to(dtype=compute_dtype)
        coefficient0 = self.coefficient0.to(dtype=compute_dtype)
        out = coefficient2 * compute_x.square() + coefficient1 * compute_x
        return (out + coefficient0).to(dtype=x.dtype)


class ProgressiveQuadraticActivation(nn.Module):
    """Blend a pretrained PReLU into its normalized quadratic student."""

    is_progressive_polynomial_activation = True

    def __init__(self, channels, input_scale=6.0, range_limit=6.0,
                 abs_quadratic_coefficient=(
                     1.0 / math.sqrt(2.0 * math.pi)),
                 stage_index=0, blend=1.0):
        super().__init__()
        if range_limit <= 0:
            raise ValueError("range_limit must be positive")
        self.prelu = nn.PReLU(channels)
        self.quadratic = NormalizedQuadratic(
            channels,
            input_scale=input_scale,
            abs_quadratic_coefficient=abs_quadratic_coefficient,
        )
        self.abs_quadratic_coefficient = float(abs_quadratic_coefficient)
        self.stage_index = int(stage_index)
        self.register_buffer(
            "blend", torch.tensor(float(blend), dtype=torch.float32))
        self.register_buffer(
            "range_limit",
            torch.tensor(float(range_limit), dtype=torch.float32))
        self._last_range_penalty = None
        self._last_distillation_loss = None
        self._last_input_absmax = None
        self._last_outside_fraction = None
        self._blend = 0.0
        self.set_blend(blend)

    def set_blend(self, blend):
        blend = float(blend)
        if not 0.0 <= blend <= 1.0:
            raise ValueError("blend must be in [0, 1]")
        self._blend = blend
        self.blend.fill_(blend)

    def range_penalty(self):
        return self._last_range_penalty

    def distillation_loss(self):
        return self._last_distillation_loss

    def range_stats(self):
        return {
            "absmax": self._last_input_absmax,
            "outside_fraction": self._last_outside_fraction,
            "blend": self.blend.detach(),
        }

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # An ordinary IResNet stores e.g. layer1.0.prelu.weight. Move that
        # tensor into the progressive wrapper while preserving strict loading.
        old_key = prefix + "weight"
        prelu_key = prefix + "prelu.weight"
        if old_key in state_dict and prelu_key not in state_dict:
            state_dict[prelu_key] = state_dict.pop(old_key)

        quadratic_prefix = prefix + "quadratic."
        has_quadratic_state = any(
            key.startswith(quadratic_prefix) for key in state_dict)
        if not has_quadratic_state:
            slope = state_dict.get(prelu_key, self.prelu.weight.detach())
            coefficient2, coefficient1, coefficient0 = (
                _prelu_initial_coefficients(
                    slope,
                    float(self.quadratic.input_scale.item()),
                    self.abs_quadratic_coefficient))
            state_dict[quadratic_prefix + "coefficient2"] = coefficient2
            state_dict[quadratic_prefix + "coefficient1"] = coefficient1
            state_dict[quadratic_prefix + "coefficient0"] = coefficient0

        # Baseline PReLU checkpoints have no progressive buffers or normalized
        # scale. Supplying their initialized values permits strict=True while
        # still rejecting partially written quadratic checkpoints.
        if not has_quadratic_state:
            for local_key, value in self.state_dict().items():
                full_key = prefix + local_key
                if full_key not in state_dict:
                    state_dict[full_key] = value.detach()

        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)
        self._blend = float(self.blend.item())

    def forward(self, x):
        if self.training:
            compute_x = (
                x.float()
                if x.dtype in (torch.float16, torch.bfloat16)
                else x
            )
            limit = self.range_limit.to(
                device=x.device, dtype=compute_x.dtype)
            excess = torch.relu(compute_x.abs() - limit)
            mean_tail = excess.square().mean()
            sample_tail = excess.flatten(1).amax(dim=1).square().mean()
            self._last_range_penalty = mean_tail + 0.1 * sample_tail
            self._last_input_absmax = compute_x.detach().abs().amax()
            self._last_outside_fraction = (
                (excess.detach() > 0).float().mean())
        else:
            self._last_range_penalty = None
            self._last_distillation_loss = None

        blend = self._blend
        if not self.training and blend <= 0.0:
            return self.prelu(x)

        prelu_out = self.prelu(x)
        quadratic_out = self.quadratic(x)
        if self.training:
            # Keep the PReLU teacher active after blend=1.  This avoids the
            # previous HerPN schedule's loss of its local target exactly when
            # an activation became fully polynomial.
            target = prelu_out.detach().float()
            self._last_distillation_loss = (
                quadratic_out.float() - target).square().mean()

        if blend <= 0.0:
            return prelu_out + quadratic_out * 0.0
        if blend >= 1.0:
            return (
                quadratic_out + prelu_out * 0.0
                if self.training else quadratic_out
            )
        return (1.0 - blend) * prelu_out + blend * quadratic_out

    def folded(self):
        if self._blend < 1.0:
            raise RuntimeError(
                "Only a fully converted quadratic activation can be folded")
        return FoldedNormalizedQuadratic.from_quadratic(self.quadratic)


class IResNet(_ProgressiveIResNet):
    """IResNet topology using progressive normalized quadratics."""

    def __init__(self, *args, quadratic_input_scale=6.0,
                 quadratic_range_limit=6.0,
                 quadratic_abs_init=1.0 / math.sqrt(2.0 * math.pi),
                 quadratic_progress=5.0, **kwargs):
        object.__setattr__(
            self, "quadratic_input_scale", float(quadratic_input_scale))
        object.__setattr__(
            self, "quadratic_abs_init", float(quadratic_abs_init))
        super().__init__(
            *args,
            herpn_range_limit=quadratic_range_limit,
            herpn_progress=quadratic_progress,
            **kwargs,
        )

    def _make_activation(self, channels, stage_name):
        return ProgressiveQuadraticActivation(
            channels=channels,
            input_scale=self.quadratic_input_scale,
            range_limit=self.herpn_range_limit,
            abs_quadratic_coefficient=self.quadratic_abs_init,
            stage_index=_STAGE_NAMES.index(stage_name),
            blend=0.0,
        )

    def progressive_activations(self):
        return [
            module for module in self.modules()
            if isinstance(module, ProgressiveQuadraticActivation)
        ]

    def set_herpn_progress(self, progress):
        """Trainer compatibility: set PReLU-to-quadratic stage progress."""
        progress = min(max(float(progress), 0.0), float(len(_STAGE_NAMES)))
        self.herpn_progress.fill_(progress)
        for activation in self.progressive_activations():
            activation.set_blend(
                min(max(progress - activation.stage_index, 0.0), 1.0))

    def set_herpn_blends(self, blends):
        """Trainer compatibility: set per-activation quadratic blends."""
        activations = {
            name: module for name, module in self.named_modules()
            if isinstance(module, ProgressiveQuadraticActivation)
        }
        unknown = sorted(set(blends).difference(activations))
        if unknown:
            raise ValueError(
                "Unknown quadratic activation names: {}".format(unknown))
        for name, activation in activations.items():
            activation.set_blend(float(blends.get(name, 0.0)))
        converted_fraction = sum(
            activation._blend for activation in activations.values()
        ) / len(activations)
        self.herpn_progress.fill_(
            converted_fraction * len(_STAGE_NAMES))

    @torch.no_grad()
    def fold_quadratic_for_inference(self):
        """Replace fully converted wrappers by exact A*x^2+B*x+C modules."""
        if self.training:
            raise RuntimeError("Call eval() before folding quadratics")
        if any(
                activation._blend < 1.0
                for activation in self.progressive_activations()):
            raise RuntimeError(
                "All activations must be fully converted before folding")

        def replace(module):
            for name, child in list(module.named_children()):
                if isinstance(child, ProgressiveQuadraticActivation):
                    setattr(module, name, child.folded())
                else:
                    replace(child)

        replace(self)
        return self

    def fold_herpn_for_inference(self):
        """Compatibility alias for tooling built around the old backbone."""
        return self.fold_quadratic_for_inference()


def _iresnet(blocks, pretrained, **kwargs):
    model = IResNet(IBasicBlock, blocks, **kwargs)
    if pretrained:
        raise ValueError("No bundled pretrained quadratic checkpoint")
    return model


def iresnet18(pretrained=False, progress=True, **kwargs):
    return _iresnet([2, 2, 2, 2], pretrained, **kwargs)


def iresnet34(pretrained=False, progress=True, **kwargs):
    return _iresnet([3, 4, 6, 3], pretrained, **kwargs)


def iresnet50(pretrained=False, progress=True, **kwargs):
    return _iresnet([3, 4, 14, 3], pretrained, **kwargs)


def iresnet100(pretrained=False, progress=True, **kwargs):
    return _iresnet([3, 13, 30, 3], pretrained, **kwargs)


def iresnet200(pretrained=False, progress=True, **kwargs):
    return _iresnet([6, 26, 60, 6], pretrained, **kwargs)

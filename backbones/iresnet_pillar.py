"""PILLAR-style polynomial iResNet backbones.

This adapts the training recipe from *Fast and Private Inference of Deep
Neural Networks by Co-designing Activation Functions* to the repository's
face-recognition iResNet topology.  Every PReLU is replaced by the paper's
p=10 quantization-aware degree-4 approximation to ReLU on ``[-5, 5]``::

    0.314453125 + 0.5*x + 0.15625*x^2 - 0.0029296875*x^4

During training only, inputs are clipped to the approximation interval after
the range penalty has been recorded.  Evaluation never clips: its encrypted
path is a fixed polynomial.  Computing ``x^2`` followed by ``x^4`` requires
two sequential ciphertext-ciphertext multiplications under FHE.
"""

import torch
from torch import nn

from .iresnet_no_relu import IBasicBlock, IResNet as _ActivationFactoryIResNet

__all__ = [
    "PILLARPolynomialReLU",
    "IResNet",
    "iresnet18",
    "iresnet34",
    "iresnet50",
    "iresnet100",
    "iresnet200",
]


_PILLAR_COEFFICIENTS = (
    322.0 / 1024.0,
    512.0 / 1024.0,
    160.0 / 1024.0,
    0.0,
    -3.0 / 1024.0,
)


class PILLARPolynomialReLU(nn.Module):
    """Fixed degree-4 ReLU approximation with training-only PILLAR guards."""

    polynomial_degree = 4
    multiplicative_depth = 2

    def __init__(self, approximation_range=5.0, regularization_range=4.8,
                 regularization_exponent=10, training_clip=True,
                 penalty_reduction="mean", penalty_tail_cap=None,
                 input_scale=1.0):
        super().__init__()
        approximation_range = float(approximation_range)
        regularization_range = float(regularization_range)
        if approximation_range != 5.0:
            raise ValueError(
                "The fixed PILLAR coefficients are fitted only on [-5, 5]; "
                "refit the polynomial before changing approximation_range")
        if not 0.0 < regularization_range <= approximation_range:
            raise ValueError(
                "regularization_range must be positive and no larger than "
                "approximation_range")
        self._validate_exponent(regularization_exponent)
        if penalty_reduction not in ("mean", "sum"):
            raise ValueError(
                "penalty_reduction must be either 'mean' or 'sum'")
        if penalty_tail_cap is not None and float(penalty_tail_cap) <= 1.0:
            raise ValueError("penalty_tail_cap must be greater than 1")
        if float(input_scale) <= 0.0:
            raise ValueError("input_scale must be positive")
        self.training_clip = bool(training_clip)
        self.penalty_reduction = str(penalty_reduction)
        self.penalty_tail_cap = (
            None if penalty_tail_cap is None else float(penalty_tail_cap))
        self._approximation_range = approximation_range
        self._regularization_exponent = int(regularization_exponent)
        self._input_scale = float(input_scale)
        self.register_buffer(
            "coefficients",
            torch.tensor(_PILLAR_COEFFICIENTS, dtype=torch.float32),
        )
        self.register_buffer(
            "approximation_range",
            torch.tensor(approximation_range, dtype=torch.float32),
        )
        self.register_buffer(
            "regularization_range",
            torch.tensor(regularization_range, dtype=torch.float32),
        )
        self.register_buffer(
            "regularization_exponent",
            torch.tensor(int(regularization_exponent), dtype=torch.int64),
        )
        self.register_buffer(
            "input_scale",
            torch.tensor(float(input_scale), dtype=torch.float32),
        )
        self._last_range_penalty = None
        self._last_input_absmax = None
        self._last_approximation_outside_fraction = None
        self._last_regularization_outside_fraction = None
        self.range_tracking = False

    @staticmethod
    def _validate_exponent(exponent):
        exponent = int(exponent)
        if exponent < 2 or exponent % 2:
            raise ValueError(
                "regularization_exponent must be an even integer >= 2")

    @torch.no_grad()
    def set_regularization_exponent(self, exponent):
        self._validate_exponent(exponent)
        self._regularization_exponent = int(exponent)
        self.regularization_exponent.fill_(int(exponent))

    @torch.no_grad()
    def set_input_scale(self, input_scale):
        input_scale = float(input_scale)
        if input_scale <= 0.0:
            raise ValueError("input_scale must be positive")
        self._input_scale = input_scale
        self.input_scale.fill_(input_scale)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # Checkpoints created before scaled PILLAR did not store this buffer.
        # Use the scale selected by the loading config while retaining strict
        # validation for every pre-existing model tensor.
        scale_key = prefix + "input_scale"
        if scale_key not in state_dict:
            state_dict[scale_key] = self.input_scale.detach().clone()
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)
        self._approximation_range = float(self.approximation_range.item())
        self._regularization_exponent = int(
            self.regularization_exponent.item())
        self._input_scale = float(self.input_scale.item())

    def range_penalty(self):
        return self._last_range_penalty

    def set_range_tracking(self, enabled):
        """Collect detached eval range statistics without changing outputs."""
        self.range_tracking = bool(enabled)

    def range_stats(self):
        return {
            "absmax": self._last_input_absmax,
            "approximation_outside_fraction": (
                self._last_approximation_outside_fraction),
            "regularization_outside_fraction": (
                self._last_regularization_outside_fraction),
            "regularization_exponent": self.regularization_exponent.detach(),
        }

    def _polynomial(self, x):
        coefficients = self.coefficients.to(device=x.device, dtype=x.dtype)
        square = x.square()
        fourth = square.square()
        return (
            coefficients[0]
            + coefficients[1] * x
            + coefficients[2] * square
            + coefficients[4] * fourth
        )

    def _element_range_penalty(self, normalized):
        """Return the paper penalty with an optional finite linear tail.

        The released implementation directly evaluates ``z**gamma``. That is
        exact and remains the default. Face iResNet occasionally produces a
        very large but finite internal outlier which overflows that training
        loss before the following activation clip can contain it. When a tail
        cap is configured, preserve the exact power through the cap and use
        its tangent line beyond it. The value and first derivative are
        continuous, extreme inputs still receive a restoring gradient, and
        this training-only guard never enters the inference graph.
        """
        magnitude = normalized.abs()
        if self.penalty_tail_cap is None:
            return magnitude.pow(self._regularization_exponent)
        cap = self.penalty_tail_cap
        exponent = self._regularization_exponent
        capped = magnitude.clamp(max=cap)
        power = capped.pow(exponent)
        slope = exponent * cap ** (exponent - 1)
        return power + slope * (magnitude - capped)

    def forward(self, x):
        compute_dtype = (
            torch.float32
            if x.dtype in (torch.float16, torch.bfloat16)
            else x.dtype
        )
        compute_x = x.to(dtype=compute_dtype)
        input_scale = self.input_scale.to(
            device=x.device, dtype=compute_dtype)
        scaled_x = compute_x / input_scale
        if self.training:
            regularization_range = self.regularization_range.to(
                device=x.device, dtype=compute_dtype)
            normalized = scaled_x / regularization_range
            element_penalty = self._element_range_penalty(normalized)
            if self.penalty_reduction == "sum":
                # PILLAR-ESPN flattens each activation and takes its L1 norm.
                # Since gamma is even, that is exactly this unnormalized sum.
                self._last_range_penalty = element_penalty.sum()
            else:
                self._last_range_penalty = element_penalty.mean()
        else:
            self._last_range_penalty = None

        if self.training or self.range_tracking:
            regularization_range = self.regularization_range.to(
                device=x.device, dtype=compute_dtype)
            detached_abs = compute_x.detach().abs()
            approximation_range = self.approximation_range.to(
                device=x.device, dtype=compute_dtype) * input_scale
            effective_regularization_range = regularization_range * input_scale
            self._last_input_absmax = detached_abs.amax()
            self._last_approximation_outside_fraction = (
                detached_abs > approximation_range).float().mean()
            self._last_regularization_outside_fraction = (
                detached_abs > effective_regularization_range).float().mean()
        else:
            self._last_input_absmax = None
            self._last_approximation_outside_fraction = None
            self._last_regularization_outside_fraction = None

        if self.training:
            # PILLAR clips only after computing the penalty and removes this
            # operation entirely from the inference graph.
            if self.training_clip:
                scaled_x = torch.clamp(
                    scaled_x,
                    min=-self._approximation_range,
                    max=self._approximation_range,
                )
        return (input_scale * self._polynomial(scaled_x)).to(dtype=x.dtype)

    def extra_repr(self):
        return (
            f"target=ReLU, interval=[-"
            f"{self._approximation_range * self._input_scale:g}, "
            f"{self._approximation_range * self._input_scale:g}], degree=4, "
            f"multiplicative_depth=2, penalty_reduction="
            f"{self.penalty_reduction}, penalty_tail_cap="
            f"{self.penalty_tail_cap}"
        )


class IResNet(_ActivationFactoryIResNet):
    """Face-recognition iResNet using the fixed PILLAR polynomial."""

    def __init__(self, *args, pillar_approximation_range=5.0,
                 pillar_regularization_range=4.8,
                 pillar_regularization_exponent=10,
                 pillar_training_clip=True,
                 pillar_penalty_reduction="mean",
                 pillar_penalty_tail_cap=None, pillar_input_scale=1.0,
                 pillar_input_scale_overrides=None, **kwargs):
        object.__setattr__(
            self, "pillar_approximation_range",
            float(pillar_approximation_range))
        object.__setattr__(
            self, "pillar_regularization_range",
            float(pillar_regularization_range))
        object.__setattr__(
            self, "pillar_regularization_exponent",
            int(pillar_regularization_exponent))
        object.__setattr__(
            self, "pillar_training_clip", bool(pillar_training_clip))
        object.__setattr__(
            self, "pillar_penalty_reduction",
            str(pillar_penalty_reduction))
        object.__setattr__(
            self, "pillar_penalty_tail_cap",
            (None if pillar_penalty_tail_cap is None
             else float(pillar_penalty_tail_cap)))
        object.__setattr__(
            self, "pillar_input_scale", float(pillar_input_scale))
        object.__setattr__(
            self, "pillar_input_scale_overrides",
            dict(pillar_input_scale_overrides or {}))
        # The parent supplies the iResNet topology and activation factory. Its
        # HerPN progress machinery sees no HerPN wrappers in this subclass.
        super().__init__(*args, herpn_progress=0.0, **kwargs)
        self.set_pillar_input_scales(self.pillar_input_scale_overrides)

    def _make_activation(self, channels, stage_name):
        del channels, stage_name
        return PILLARPolynomialReLU(
            approximation_range=self.pillar_approximation_range,
            regularization_range=self.pillar_regularization_range,
            regularization_exponent=self.pillar_regularization_exponent,
            training_clip=self.pillar_training_clip,
            penalty_reduction=self.pillar_penalty_reduction,
            penalty_tail_cap=self.pillar_penalty_tail_cap,
            input_scale=self.pillar_input_scale,
        )

    def set_pillar_input_scales(self, overrides):
        activations = {
            name: module for name, module in self.named_modules()
            if isinstance(module, PILLARPolynomialReLU)
        }
        unknown = sorted(set(overrides).difference(activations))
        if unknown:
            raise ValueError(
                f"Unknown PILLAR activation names: {unknown}")
        for name, input_scale in overrides.items():
            activations[name].set_input_scale(input_scale)

    def pillar_activations(self):
        return [
            module for module in self.modules()
            if isinstance(module, PILLARPolynomialReLU)
        ]

    def set_pillar_regularization_exponent(self, exponent):
        for activation in self.pillar_activations():
            activation.set_regularization_exponent(exponent)

    def set_pillar_range_tracking(self, enabled):
        """Enable detached range profiling for representative eval passes."""
        for activation in self.pillar_activations():
            activation.set_range_tracking(enabled)

    def pillar_range_penalty(self):
        penalties = [
            activation.range_penalty()
            for activation in self.pillar_activations()
        ]
        penalties = [penalty for penalty in penalties if penalty is not None]
        if not penalties:
            return next(self.parameters()).new_zeros(())
        # Equation (7) in the paper averages one penalty per activation layer.
        return torch.stack(penalties).mean()

    def pillar_range_stats(self):
        return {
            name: module.range_stats()
            for name, module in self.named_modules()
            if isinstance(module, PILLARPolynomialReLU)
        }

    def pillar_range_summary(self):
        stats = list(self.pillar_range_stats().values())
        absmax = [item["absmax"] for item in stats
                  if item["absmax"] is not None]
        approximation_outside = [
            item["approximation_outside_fraction"] for item in stats
            if item["approximation_outside_fraction"] is not None
        ]
        regularization_outside = [
            item["regularization_outside_fraction"] for item in stats
            if item["regularization_outside_fraction"] is not None
        ]
        zero = next(self.parameters()).new_zeros(())
        return {
            "input_absmax": torch.stack(absmax).amax() if absmax else zero,
            "approximation_outside_fraction": (
                torch.stack(approximation_outside).mean()
                if approximation_outside else zero),
            "regularization_outside_fraction": (
                torch.stack(regularization_outside).mean()
                if regularization_outside else zero),
        }


def _iresnet(block, layers, pretrained, **kwargs):
    if pretrained:
        raise ValueError("No bundled pretrained PILLAR checkpoint")
    return IResNet(block, layers, **kwargs)


def iresnet18(pretrained=False, progress=True, **kwargs):
    del progress
    return _iresnet(IBasicBlock, [2, 2, 2, 2], pretrained, **kwargs)


def iresnet34(pretrained=False, progress=True, **kwargs):
    del progress
    return _iresnet(IBasicBlock, [3, 4, 6, 3], pretrained, **kwargs)


def iresnet50(pretrained=False, progress=True, **kwargs):
    del progress
    return _iresnet(IBasicBlock, [3, 4, 14, 3], pretrained, **kwargs)


def iresnet100(pretrained=False, progress=True, **kwargs):
    del progress
    return _iresnet(IBasicBlock, [3, 13, 30, 3], pretrained, **kwargs)


def iresnet200(pretrained=False, progress=True, **kwargs):
    del progress
    return _iresnet(IBasicBlock, [6, 26, 60, 6], pretrained, **kwargs)

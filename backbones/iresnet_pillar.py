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
                 penalty_reduction="mean"):
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
        self.training_clip = bool(training_clip)
        self.penalty_reduction = str(penalty_reduction)
        self._approximation_range = approximation_range
        self._regularization_exponent = int(regularization_exponent)
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

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)
        self._approximation_range = float(self.approximation_range.item())
        self._regularization_exponent = int(
            self.regularization_exponent.item())

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

    def forward(self, x):
        compute_dtype = (
            torch.float32
            if x.dtype in (torch.float16, torch.bfloat16)
            else x.dtype
        )
        compute_x = x.to(dtype=compute_dtype)
        if self.training:
            regularization_range = self.regularization_range.to(
                device=x.device, dtype=compute_dtype)
            normalized = compute_x / regularization_range
            element_penalty = normalized.pow(
                self._regularization_exponent)
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
                device=x.device, dtype=compute_dtype)
            self._last_input_absmax = detached_abs.amax()
            self._last_approximation_outside_fraction = (
                detached_abs > approximation_range).float().mean()
            self._last_regularization_outside_fraction = (
                detached_abs > regularization_range).float().mean()
        else:
            self._last_input_absmax = None
            self._last_approximation_outside_fraction = None
            self._last_regularization_outside_fraction = None

        if self.training:
            # PILLAR clips only after computing the penalty and removes this
            # operation entirely from the inference graph.
            if self.training_clip:
                compute_x = torch.clamp(
                    compute_x,
                    min=-self._approximation_range,
                    max=self._approximation_range,
                )
        return self._polynomial(compute_x).to(dtype=x.dtype)

    def extra_repr(self):
        return (
            f"target=ReLU, interval=[-{self._approximation_range:g}, "
            f"{self._approximation_range:g}], degree=4, "
            f"multiplicative_depth=2, penalty_reduction="
            f"{self.penalty_reduction}"
        )


class IResNet(_ActivationFactoryIResNet):
    """Face-recognition iResNet using the fixed PILLAR polynomial."""

    def __init__(self, *args, pillar_approximation_range=5.0,
                 pillar_regularization_range=4.8,
                 pillar_regularization_exponent=10,
                 pillar_training_clip=True,
                 pillar_penalty_reduction="mean", **kwargs):
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
        # The parent supplies the iResNet topology and activation factory. Its
        # HerPN progress machinery sees no HerPN wrappers in this subclass.
        super().__init__(*args, herpn_progress=0.0, **kwargs)

    def _make_activation(self, channels, stage_name):
        del channels, stage_name
        return PILLARPolynomialReLU(
            approximation_range=self.pillar_approximation_range,
            regularization_range=self.pillar_regularization_range,
            regularization_exponent=self.pillar_regularization_exponent,
            training_clip=self.pillar_training_clip,
            penalty_reduction=self.pillar_penalty_reduction,
        )

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

"""Polynomial activations used by the stability analysis workflow."""

import math

import torch
from torch import nn


class PolynomialActivation(nn.Module):
    """Scalar polynomial evaluated by Horner's rule.

    Coefficients are ordered from constant to highest degree.  ``interval``
    documents the interval on which the polynomial was designed; it does not
    clamp its input because clamping is not polynomial/FHE-friendly.
    """

    def __init__(self, coefficients, interval=(-6.0, 6.0), target="user-defined"):
        super().__init__()
        if not coefficients:
            raise ValueError("at least one coefficient is required")
        low, high = map(float, interval)
        if not low < high:
            raise ValueError("interval must satisfy low < high")
        self.register_buffer(
            "coefficients", torch.as_tensor(coefficients, dtype=torch.float64))
        self.interval = (low, high)
        self.target = str(target)

    @property
    def degree(self):
        return self.coefficients.numel() - 1

    def forward(self, x):
        coefficients = self.coefficients.to(device=x.device, dtype=x.dtype)
        result = torch.zeros_like(x) + coefficients[-1]
        for coefficient in coefficients[:-1].flip(0):
            result = result * x + coefficient
        return result


class HerPN(PolynomialActivation):
    """Default Gaussian-Hermite quadratic approximation to PReLU.

    Approximation target on ``[-input_scale, input_scale]`` is PReLU with
    negative slope ``prelu_slope``.  It uses
    ``abs(x) ~= (x^2 + 1)/sqrt(2*pi)`` under a standard-normal input
    weighting. ``input_scale`` is the monitored safety/approximation interval,
    not a claim of a uniform minimax fit over that interval.
    Thus the encrypted path is ``A*x^2 + B*x + C`` and has one sequential
    ciphertext-ciphertext multiplication.
    """

    def __init__(self, input_scale=6.0, prelu_slope=0.25):
        scale = float(input_scale)
        slope = float(prelu_slope)
        if scale <= 0:
            raise ValueError("input_scale must be positive")
        even = 0.5 * (1.0 - slope)
        hermite = 1.0 / math.sqrt(2.0 * math.pi)
        coefficients = (
            hermite * even,
            0.5 * (1.0 + slope),
            hermite * even,
        )
        super().__init__(
            coefficients,
            interval=(-scale, scale),
            target=f"PReLU(negative_slope={slope:g})",
        )


def make_herpn_for(module, input_scale=6.0):
    """Create the default HerPN approximation for a replaced activation."""
    slope = 0.0
    target = "ReLU"
    if isinstance(module, nn.PReLU):
        if module.num_parameters != 1:
            # The analyzer intentionally uses the mean slope for a scalar,
            # model-agnostic polynomial. Per-channel custom modules remain
            # available through --activation-file.
            slope = float(module.weight.detach().float().mean())
        else:
            slope = float(module.weight.detach().item())
        target = f"PReLU(mean_negative_slope={slope:g})"
    elif isinstance(module, nn.GELU):
        # HerPN is an even/linear Hermite approximation.  Its default GELU
        # proxy uses the ReLU limit, which GELU smoothly approximates.
        target = "GELU (via ReLU Hermite proxy)"
    activation = HerPN(input_scale=input_scale, prelu_slope=slope)
    activation.target = target
    return activation

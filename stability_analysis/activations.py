"""Polynomial activations used by the stability analysis workflow."""

import math

import torch
from torch import nn
from torch.nn import functional as F


class PolynomialActivation(nn.Module):
    """Scalar polynomial evaluated by Horner's rule.

    Coefficients are ordered from constant to highest degree.  ``interval``
    documents the interval on which the polynomial was designed; it does not
    clamp its input because clamping is not polynomial/FHE-friendly.
    """

    def __init__(self, coefficients, interval=(-6.0, 6.0), target="user-defined"):
        super().__init__()
        if coefficients is None or len(coefficients) == 0:
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

    def evaluate_flat(self, values, channels=None):
        coefficients = self.coefficients.to(
            device=values.device, dtype=values.dtype)
        result = torch.zeros_like(values) + coefficients[-1]
        for coefficient in coefficients[:-1].flip(0):
            result = result * values + coefficient
        return result

    def derivative_flat(self, values, channels=None):
        coefficients = self.coefficients.to(
            device=values.device, dtype=values.dtype)
        if coefficients.numel() == 1:
            return torch.zeros_like(values)
        result = torch.zeros_like(values) + (
            (coefficients.numel() - 1) * coefficients[-1])
        for degree in range(coefficients.numel() - 2, 0, -1):
            result = result * values + degree * coefficients[degree]
        return result


class ChannelwisePolynomialActivation(nn.Module):
    """Channel-wise scalar polynomials for convolutional activations."""

    def __init__(self, coefficients, interval=(-6.0, 6.0),
                 target="user-defined"):
        super().__init__()
        coefficients = torch.as_tensor(coefficients, dtype=torch.float64)
        if coefficients.ndim != 2 or coefficients.shape[1] == 0:
            raise ValueError(
                "coefficients must have shape [degree + 1, channels]")
        low, high = map(float, interval)
        if not low < high:
            raise ValueError("interval must satisfy low < high")
        self.register_buffer("coefficients", coefficients)
        self.interval = (low, high)
        self.target = str(target)

    @property
    def degree(self):
        return self.coefficients.shape[0] - 1

    def _broadcast(self, coefficients, x):
        shape = [1, coefficients.numel()] + [1] * (x.ndim - 2)
        return coefficients.reshape(shape).to(device=x.device, dtype=x.dtype)

    def forward(self, x):
        coefficients = self.coefficients
        result = torch.zeros_like(x) + self._broadcast(coefficients[-1], x)
        for coefficient in coefficients[:-1].flip(0):
            result = result * x + self._broadcast(coefficient, x)
        return result

    def _select(self, coefficients, values, channels):
        if channels is None:
            raise ValueError("channels are required for channel-wise evaluation")
        return coefficients.to(
            device=values.device, dtype=values.dtype)[channels.long()]

    def evaluate_flat(self, values, channels=None):
        coefficients = self.coefficients
        result = torch.zeros_like(values) + self._select(
            coefficients[-1], values, channels)
        for coefficient in coefficients[:-1].flip(0):
            result = result * values + self._select(
                coefficient, values, channels)
        return result

    def derivative_flat(self, values, channels=None):
        coefficients = self.coefficients
        if coefficients.shape[0] == 1:
            return torch.zeros_like(values)
        result = torch.zeros_like(values) + self._select(
            (coefficients.shape[0] - 1) * coefficients[-1],
            values, channels)
        for degree in range(coefficients.shape[0] - 2, 0, -1):
            result = result * values + self._select(
                degree * coefficients[degree], values, channels)
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
        slopes = module.weight.detach().double().flatten()
        if module.num_parameters != 1:
            even = 0.5 * (1.0 - slopes)
            hermite = 1.0 / math.sqrt(2.0 * math.pi)
            coefficients = torch.stack((
                hermite * even,
                0.5 * (1.0 + slopes),
                hermite * even,
            ))
            return ChannelwisePolynomialActivation(
                coefficients,
                interval=(-float(input_scale), float(input_scale)),
                target=(
                    "channel-wise PReLU Gaussian-Hermite proxy "
                    f"(mean_negative_slope={float(slopes.mean()):g})"
                ),
            )
        slope = float(slopes.item())
        target = f"PReLU(negative_slope={slope:g})"
    elif isinstance(module, nn.GELU):
        # HerPN is an even/linear Hermite approximation.  Its default GELU
        # proxy uses the ReLU limit, which GELU smoothly approximates.
        target = "GELU (via ReLU Hermite proxy)"
    activation = HerPN(input_scale=input_scale, prelu_slope=slope)
    activation.target = target
    return activation


def make_uniform_quadratic_for(module, input_scale=6.0):
    """Return the uniform-L2 quadratic fit on a symmetric finite interval.

    For PReLU/ReLU this uses the analytic least-squares projection of ``|x|``
    onto ``span(1, x^2)`` on ``[-input_scale, input_scale]``. GELU is fitted
    numerically on a dense uniform grid. No clipping is performed.
    """
    scale = float(input_scale)
    if scale <= 0:
        raise ValueError("input_scale must be positive")

    if isinstance(module, nn.PReLU):
        slopes = module.weight.detach().double().flatten()
        constant = (1.0 - slopes) * (3.0 * scale / 32.0)
        linear = 0.5 * (1.0 + slopes)
        quadratic = (1.0 - slopes) * (15.0 / (32.0 * scale))
        coefficients = torch.stack((constant, linear, quadratic))
        target = (
            "channel-wise PReLU uniform-L2 fit"
            if slopes.numel() > 1 else "PReLU uniform-L2 fit"
        )
        if slopes.numel() > 1:
            return ChannelwisePolynomialActivation(
                coefficients, interval=(-scale, scale), target=target)
        return PolynomialActivation(
            coefficients[:, 0], interval=(-scale, scale), target=target)

    if isinstance(module, nn.ReLU):
        coefficients = [3.0 * scale / 32.0, 0.5, 15.0 / (32.0 * scale)]
        return PolynomialActivation(
            coefficients, interval=(-scale, scale),
            target="ReLU uniform-L2 fit")

    if isinstance(module, nn.GELU):
        grid = torch.linspace(-scale, scale, 4097, dtype=torch.float64)
        design = torch.stack((torch.ones_like(grid), grid, grid.square()), 1)
        target = F.gelu(grid)
        solution = torch.pinverse(design).mv(target)
        return PolynomialActivation(
            solution, interval=(-scale, scale),
            target="GELU uniform-L2 fit")

    raise TypeError("unsupported activation type: {}".format(type(module)))


def evaluate_target_flat(module, values, channels=None):
    """Evaluate a supported teacher activation on sampled scalar values."""
    if isinstance(module, nn.PReLU):
        slopes = module.weight.detach().to(
            device=values.device, dtype=values.dtype).flatten()
        if slopes.numel() == 1:
            slope = slopes[0]
        else:
            if channels is None:
                raise ValueError("channels are required for channel-wise PReLU")
            slope = slopes[channels.long()]
        return torch.where(values >= 0, values, slope * values)
    if isinstance(module, nn.ReLU):
        return values.clamp_min(0)
    if isinstance(module, nn.GELU):
        return F.gelu(values)
    raise TypeError("unsupported activation type: {}".format(type(module)))


def evaluate_target_derivative_flat(module, values, channels=None):
    """Evaluate the teacher's scalar derivative away from zero."""
    if isinstance(module, nn.PReLU):
        slopes = module.weight.detach().to(
            device=values.device, dtype=values.dtype).flatten()
        if slopes.numel() == 1:
            slope = slopes[0]
        else:
            if channels is None:
                raise ValueError("channels are required for channel-wise PReLU")
            slope = slopes[channels.long()]
        return torch.where(values >= 0, torch.ones_like(values), slope)
    if isinstance(module, nn.ReLU):
        return (values > 0).to(values.dtype)
    if isinstance(module, nn.GELU):
        root_two = math.sqrt(2.0)
        root_two_pi = math.sqrt(2.0 * math.pi)
        return (
            0.5 * (1.0 + torch.erf(values / root_two))
            + values * torch.exp(-0.5 * values.square()) / root_two_pi
        )
    raise TypeError("unsupported activation type: {}".format(type(module)))

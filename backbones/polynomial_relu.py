"""Fixed polynomial approximations to ReLU on a public interval.

The input scale ``S`` defines the approximation target ``ReLU(x)`` on
``[-S, S]``.  Scaling by ``S`` and ``1/S`` is multiplication by public
plaintext constants in an FHE implementation.
"""

import torch
from torch import nn

__all__ = ["ChebyReLU", "PreciseReLUAlpha10"]


def _compute_dtype(x):
    return torch.float32 if x.dtype in (torch.float16, torch.bfloat16) else x.dtype


def _polyval(x, coefficients):
    result = coefficients[-1].to(dtype=x.dtype, device=x.device)
    for coefficient in coefficients[:-1].flip(0):
        result = result * x + coefficient.to(dtype=x.dtype, device=x.device)
    return result


def _polyval_with_derivative(x, coefficients):
    """Evaluate a power-basis polynomial and its derivative by Horner."""
    value = coefficients[-1].to(dtype=x.dtype, device=x.device)
    derivative = torch.zeros((), dtype=x.dtype, device=x.device)
    for coefficient in coefficients[:-1].flip(0):
        derivative = derivative * x + value
        value = value * x + coefficient.to(dtype=x.dtype, device=x.device)
    return value, derivative


class _PreciseReLUAlpha10Function(torch.autograd.Function):
    """Memory-efficient Alpha10 with analytical recomputation in backward.

    Ordinary autograd retains every full-sized Horner intermediate from the
    degree-7, degree-7, and degree-13 composition.  R50 applies the function
    25 times, so those saved tensors dominate GPU memory.  The coefficients
    are fixed public constants: save only the activation input and recompute
    the polynomial values and derivatives during backward instead.
    """

    @staticmethod
    def forward(ctx, x, p1_coeffs, p2_coeffs, p3_coeffs, input_scale):
        compute_dtype = _compute_dtype(x)
        compute_x = x.to(dtype=compute_dtype)
        scale = input_scale.to(device=x.device, dtype=compute_dtype)
        p1 = p1_coeffs.to(device=x.device, dtype=compute_dtype)
        p2 = p2_coeffs.to(device=x.device, dtype=compute_dtype)
        p3 = p3_coeffs.to(device=x.device, dtype=compute_dtype)
        scaled_x = compute_x * scale.reciprocal()
        p1_x = _polyval(scaled_x, p1)
        p2_x = _polyval(p1_x, p2)
        p3_x = _polyval(p2_x, p3)
        output = scale * 0.5 * (scaled_x + scaled_x * p3_x)
        ctx.save_for_backward(
            x, p1_coeffs, p2_coeffs, p3_coeffs, input_scale)
        return output.to(dtype=x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        x, p1_coeffs, p2_coeffs, p3_coeffs, input_scale = (
            ctx.saved_tensors)
        compute_dtype = _compute_dtype(x)
        compute_x = x.to(dtype=compute_dtype)
        scale = input_scale.to(device=x.device, dtype=compute_dtype)
        p1 = p1_coeffs.to(device=x.device, dtype=compute_dtype)
        p2 = p2_coeffs.to(device=x.device, dtype=compute_dtype)
        p3 = p3_coeffs.to(device=x.device, dtype=compute_dtype)

        scaled_x = compute_x * scale.reciprocal()
        p1_x, p1_derivative = _polyval_with_derivative(scaled_x, p1)
        p2_x, p2_derivative = _polyval_with_derivative(p1_x, p2)
        p3_x, p3_derivative = _polyval_with_derivative(p2_x, p3)
        composed_derivative = (
            p3_derivative * p2_derivative * p1_derivative)
        input_derivative = 0.5 * (
            1.0 + p3_x + scaled_x * composed_derivative)
        grad_input = grad_output.to(dtype=compute_dtype) * input_derivative
        return grad_input.to(dtype=x.dtype), None, None, None, None


def _cheby_relu_forward(x, coefficients, scale, degree):
    z = x * scale.reciprocal()
    z_squared = z * z
    z_fourth = z_squared * z_squared
    even_part = coefficients[0] * z_squared + coefficients[1] * z_fourth
    if degree >= 8:
        z_sixth = z_fourth * z_squared
        z_eighth = z_fourth * z_fourth
        even_part = (
            even_part
            + coefficients[2] * z_sixth
            + coefficients[3] * z_eighth
        )
    if degree == 16:
        z_tenth = z_eighth * z_squared
        z_twelfth = z_eighth * z_fourth
        z_fourteenth = z_eighth * z_sixth
        z_sixteenth = z_eighth * z_eighth
        even_part = (
            even_part
            + coefficients[4] * z_tenth
            + coefficients[5] * z_twelfth
            + coefficients[6] * z_fourteenth
            + coefficients[7] * z_sixteenth
        )
    return 0.5 * x + scale * even_part


class _ChebyReLUFunction(torch.autograd.Function):
    """Fixed ChebyReLU with input-only activation storage."""

    @staticmethod
    def forward(ctx, x, coefficients, input_scale, degree):
        compute_dtype = _compute_dtype(x)
        compute_x = x.to(dtype=compute_dtype)
        scale = input_scale.to(device=x.device, dtype=compute_dtype)
        compute_coefficients = coefficients.to(
            device=x.device, dtype=compute_dtype)
        output = _cheby_relu_forward(
            compute_x, compute_coefficients, scale, int(degree))
        ctx.degree = int(degree)
        ctx.save_for_backward(x, coefficients, input_scale)
        return output.to(dtype=x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        x, coefficients, input_scale = ctx.saved_tensors
        compute_dtype = _compute_dtype(x)
        compute_x = x.to(dtype=compute_dtype)
        scale = input_scale.to(device=x.device, dtype=compute_dtype)
        compute_coefficients = coefficients.to(
            device=x.device, dtype=compute_dtype)
        z = compute_x * scale.reciprocal()
        z_squared = z.square()

        # q(z)=0.5*z+sum_k c_k*z^(2k), so dy/dx=q'(z); the
        # outer S and inner 1/S public scales cancel exactly.
        term_count = compute_coefficients.numel()
        even_derivative = (
            2.0 * term_count * compute_coefficients[-1])
        for index in range(term_count - 2, -1, -1):
            power = index + 1
            even_derivative = (
                even_derivative * z_squared
                + 2.0 * power * compute_coefficients[index]
            )
        input_derivative = 0.5 + z * even_derivative
        grad_input = grad_output.to(dtype=compute_dtype) * input_derivative
        return grad_input.to(dtype=x.dtype), None, None, None


class ChebyReLU(nn.Module):
    """Zero-preserving minimax approximation to ReLU on ``[-S, S]``."""

    _NORMALIZED_POWER_COEFFS = {
        4: (1.05146222424, -0.581234022404),
        8: (
            2.3251649858241135,
            -7.139218135121423,
            9.889731805079089,
            -4.603662092314254,
        ),
        16: (
            4.637380568115935,
            -59.60499617236257,
            412.04080125849953,
            -1505.0765135944926,
            3072.4113964565195,
            -3524.8808527426254,
            2123.263135406183,
            -522.3044135526843,
        ),
    }
    _MULTIPLICATIVE_DEPTHS = {4: 2, 8: 3, 16: 4}

    def __init__(self, input_scale=8.0, degree=4):
        super().__init__()
        if input_scale <= 0:
            raise ValueError("input_scale must be positive")
        if degree not in self._NORMALIZED_POWER_COEFFS:
            raise ValueError("ChebyReLU degree must be 4, 8, or 16")
        self.degree = int(degree)
        self.multiplicative_depth = self._MULTIPLICATIVE_DEPTHS[self.degree]
        self.register_buffer(
            "input_scale", torch.tensor(float(input_scale), dtype=torch.float32))
        self.register_buffer(
            "normalized_power_coeffs",
            torch.tensor(
                self._NORMALIZED_POWER_COEFFS[self.degree],
                dtype=torch.float32))

    def forward(self, x):
        return _ChebyReLUFunction.apply(
            x,
            self.normalized_power_coeffs,
            self.input_scale,
            self.degree,
        )


class PreciseReLUAlpha10(nn.Module):
    """Alpha-10 composite approximation to ReLU on ``[-S, S]``.

    This is Appendix A's ``0.5*x*(1 + (p3 o p2 o p1)(x))`` construction
    from *Precise Approximation of Convolutional Neural Networks*.  Its
    component degrees are 7, 7, and 13; the resulting ReLU polynomial has
    algebraic degree 638.  It is therefore the accurate curriculum teacher,
    not the intended low-depth final FHE activation.
    """

    component_degrees = (7, 7, 13)
    algebraic_degree = 638

    _P1_COEFFS = (
        -1.68048812248597e-47, 1.08541842577442e1,
        5.19213405604261e-46, -6.22833925211098e1,
        -1.67358715007438e-45, 1.14369227820443e2,
        1.15437076692363e-45, -6.28023496973074e1,
    )
    _P2_COEFFS = (
        7.86253562483970e-39, 4.13976170985111,
        -7.18241741649940e-38, -5.84997640211679,
        5.17878634442782e-38, 2.94376255659280,
        -9.33059743960049e-39, -4.54530437460152e-1,
    )
    _P3_COEFFS = (
        3.75374153583292e-39, 3.29956739043733,
        -1.04537140020889e-37, -7.84227260291355,
        4.18647895984231e-37, 1.28907764115564e1,
        -6.09510159540855e-37, -1.24917112584486e1,
        4.05475441247124e-37, 6.94167991428074,
        -1.26770087815848e-37, -2.04298067399942,
        1.52452197400636e-38, 2.46407138926031e-1,
    )

    def __init__(self, input_scale=1.0):
        super().__init__()
        if input_scale <= 0:
            raise ValueError("input_scale must be positive")
        self.register_buffer(
            "input_scale", torch.tensor(float(input_scale), dtype=torch.float32))
        self.register_buffer(
            "p1_coeffs", torch.tensor(self._P1_COEFFS, dtype=torch.float32))
        self.register_buffer(
            "p2_coeffs", torch.tensor(self._P2_COEFFS, dtype=torch.float32))
        self.register_buffer(
            "p3_coeffs", torch.tensor(self._P3_COEFFS, dtype=torch.float32))

    def forward(self, x):
        return _PreciseReLUAlpha10Function.apply(
            x,
            self.p1_coeffs,
            self.p2_coeffs,
            self.p3_coeffs,
            self.input_scale,
        )

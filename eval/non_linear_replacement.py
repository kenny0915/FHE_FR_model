# coding: utf-8

import torch


class _THORPolynomialGELUFunction(torch.autograd.Function):
    @staticmethod
    def _polyval(x, coeffs):
        y = coeffs[-1].to(dtype=x.dtype, device=x.device)
        for coeff in coeffs[:-1].flip(0):
            y = y * x + coeff.to(dtype=x.dtype, device=x.device)
        return y

    @staticmethod
    def forward(ctx, x, p1_coeffs, p2_coeffs, input_scale):
        compute_x = x.float()
        p1 = p1_coeffs.float()
        p2 = p2_coeffs.float()
        scaled_x = compute_x / float(input_scale)
        p1_x = _THORPolynomialGELUFunction._polyval(scaled_x, p1)
        tanh_half = _THORPolynomialGELUFunction._polyval(p1_x, p2)
        return (compute_x * (0.5 + tanh_half)).to(dtype=x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        raise RuntimeError("THORPolynomialGELU is intended for inference in eval scripts.")


class THORPolynomialGELU(torch.nn.Module):
    _P1_COEFFS = (
        -1.06240033e-05, 1.64454894e-04, -5.83533517e-04, -3.80912692e-04,
        2.24431193e-03, 8.92295204e-03, -1.05277477e-02, -1.91827040e-02,
        -2.04634786e-01, 4.54014410e-01, -5.40759203e-01, 5.67745523e+00,
        -1.36433727e+01, 1.82574621e+01, -8.48849601e+01, 1.28686741e+02,
        3.66720281e+02, -1.01400159e+03, -1.26278856e+02, 2.21728878e+03,
        -9.95421415e+02, -2.31059465e+03, 1.73583957e+03, 1.27394360e+03,
        -1.27836230e+03, -3.66781716e+02, 4.79663919e+02, 4.94610178e+01,
        -9.06754761e+01, -2.36515790e+00, 8.74311855e+00, 1.62838703e-02,
    )
    _P2_COEFFS = (
        -1.70270667e+02, 6.81076279e+01, 1.79197364e+03, -6.81621043e+02,
        -8.49256169e+03, 3.05629446e+03, 2.39579397e+04, -8.10435126e+03,
        -4.48145152e+04, 1.41297616e+04, 5.86197512e+04, -1.70371505e+04,
        -5.51326382e+04, 1.45532495e+04, 3.77866438e+04, -8.87673890e+03,
        -1.89514802e+04, 3.84972853e+03, 6.94169727e+03, -1.16901058e+03,
        -1.84658407e+03, 2.41693754e+02, 3.54452276e+02, -3.24499570e+01,
        -4.91918227e+01, 2.58122977e+00, 5.78392852e+00, -9.45171527e-02,
    )

    def __init__(self, input_scale=64.0):
        super().__init__()
        self.input_scale = float(input_scale)
        self.register_buffer("p1_coeffs", torch.tensor(tuple(reversed(self._P1_COEFFS)), dtype=torch.float32))
        self.register_buffer("p2_coeffs", 0.5 * torch.tensor(tuple(reversed(self._P2_COEFFS)), dtype=torch.float32))

    def forward(self, x):
        return _THORPolynomialGELUFunction.apply(
            x, self.p1_coeffs, self.p2_coeffs, self.input_scale
        )


class ChebyReLU(torch.nn.Module):
    _NORMALIZED_POWER_COEFFS = (1.05146222424, -0.581234022404)

    def __init__(self, input_scale=8.0):
        super().__init__()
        if input_scale <= 0:
            raise ValueError('input_scale must be positive')
        self.register_buffer(
            'input_scale', torch.tensor(float(input_scale), dtype=torch.float32))
        self.register_buffer(
            'normalized_power_coeffs',
            torch.tensor(self._NORMALIZED_POWER_COEFFS, dtype=torch.float32))

    def forward(self, x):
        compute_dtype = (
            torch.float32
            if x.dtype in (torch.float16, torch.bfloat16)
            else x.dtype
        )
        compute_x = x.to(dtype=compute_dtype)
        scale = self.input_scale.to(device=x.device, dtype=compute_dtype)
        coefficients = self.normalized_power_coeffs.to(
            device=x.device, dtype=compute_dtype)

        z = compute_x / scale
        z_squared = z * z
        z_fourth = z_squared * z_squared
        even_part = coefficients[0] * z_squared + coefficients[1] * z_fourth
        out = 0.5 * compute_x + scale * even_part
        return out.to(dtype=x.dtype)


class PreciseReLUAlpha10(torch.nn.Module):
    # Appendix A of "Precise Approximation of Convolutional Neural Networks":
    # r_alpha(x) = 0.5 * (x + x * (p3 o p2 o p1)(x)), alpha = 10.
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
        self.input_scale = float(input_scale)
        self.register_buffer("p1_coeffs", torch.tensor(self._P1_COEFFS, dtype=torch.float32))
        self.register_buffer("p2_coeffs", torch.tensor(self._P2_COEFFS, dtype=torch.float32))
        self.register_buffer("p3_coeffs", torch.tensor(self._P3_COEFFS, dtype=torch.float32))

    @staticmethod
    def _polyval(x, coeffs):
        y = coeffs[-1].to(dtype=x.dtype, device=x.device)
        for coeff in coeffs[:-1].flip(0):
            y = y * x + coeff.to(dtype=x.dtype, device=x.device)
        return y

    def forward(self, x):
        compute_x = x.float()
        scaled_x = compute_x / self.input_scale
        p1_x = self._polyval(scaled_x, self.p1_coeffs.float())
        p2_x = self._polyval(p1_x, self.p2_coeffs.float())
        p3_x = self._polyval(p2_x, self.p3_coeffs.float())
        out = self.input_scale * 0.5 * (scaled_x + scaled_x * p3_x)
        return out.to(dtype=x.dtype)


class PReLU_Approx(torch.nn.Module):
    def __init__(self, slope, input_scale=1.0):
        super().__init__()
        slope = slope.detach().clone().float().reshape(-1)
        # self.relu = PreciseReLUAlpha10(input_scale=input_scale)
        self.relu = ChebyReLU(input_scale=input_scale)
        self.register_buffer("slope", slope)

    def _slope_for(self, x):
        slope = self.slope.to(dtype=x.dtype, device=x.device)
        if slope.numel() == 1:
            return slope.reshape(1)
        if x.ndim >= 2 and slope.numel() == x.shape[1]:
            return slope.reshape(1, slope.numel(), *([1] * (x.ndim - 2)))
        return slope

    def forward(self, x):
        slope = self._slope_for(x)
        return slope * x + (1 - slope) * self.relu(x)


def replace_resnet_activations_with_poly(module, input_scale=1.0):
    replaced = 0
    for name, child in module.named_children():
        if isinstance(child, torch.nn.PReLU):
            setattr(module, name, PReLU_Approx(child.weight, input_scale=input_scale))
            replaced += 1
        else:
            replaced += replace_resnet_activations_with_poly(child, input_scale=input_scale)
    return replaced


def replace_poolformer_gelu_with_thor(module):
    replaced = 0
    for name, child in module.named_children():
        if isinstance(child, THORPolynomialGELU):
            continue
        if isinstance(child, torch.nn.GELU):
            setattr(module, name, THORPolynomialGELU())
            replaced += 1
        else:
            replaced += replace_poolformer_gelu_with_thor(child)
    return replaced

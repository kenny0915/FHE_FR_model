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
    """Zero-preserving polynomial approximation to ReLU on ``[-scale, scale]``.

    For ``z = x / scale``, both variants have the normalized form

        q(z) = 0.5*z + sum(c_2k * z^(2k)), 1 <= k <= degree/2.

    Degrees 8 and 16 use constrained minimax fits to ReLU on ``[-1, 1]``.
    Their normalized maximum absolute errors are about 0.02798 and 0.01406,
    respectively, versus about 0.06159 for the retained degree-4
    approximation.  Balanced power schedules keep multiplicative depth at 2,
    3, and 4 for degrees 4, 8, and 16.
    """
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
            raise ValueError('input_scale must be positive')
        if degree not in self._NORMALIZED_POWER_COEFFS:
            raise ValueError('ChebyReLU degree must be 4, 8, or 16')
        self.degree = int(degree)
        self.multiplicative_depth = self._MULTIPLICATIVE_DEPTHS[self.degree]
        self.register_buffer(
            'input_scale', torch.tensor(float(input_scale), dtype=torch.float32))
        self.register_buffer(
            'normalized_power_coeffs',
            torch.tensor(
                self._NORMALIZED_POWER_COEFFS[self.degree],
                dtype=torch.float32))

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
        if self.degree >= 8:
            z_sixth = z_fourth * z_squared
            z_eighth = z_fourth * z_fourth
            even_part = (
                even_part
                + coefficients[2] * z_sixth
                + coefficients[3] * z_eighth
            )
        if self.degree == 16:
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
        out = 0.5 * compute_x + scale * even_part
        return out.to(dtype=x.dtype)


class _PreciseReLU(torch.nn.Module):
    """Scaled Appendix-A PreciseReLU on ``[-input_scale, input_scale]``."""

    alpha = None
    component_degrees = ()
    multiplicative_depth = None
    non_scalar_multiplications = None
    _COMPONENT_COEFFS = ()

    def __init__(self, input_scale=1.0):
        super().__init__()
        if input_scale <= 0:
            raise ValueError("input_scale must be positive")
        self.input_scale = float(input_scale)
        for index, coefficients in enumerate(self._COMPONENT_COEFFS, start=1):
            self.register_buffer(
                "p{}_coeffs".format(index),
                torch.tensor(coefficients, dtype=torch.float32),
            )

    @staticmethod
    def _polyval(x, coeffs):
        y = coeffs[-1].to(dtype=x.dtype, device=x.device)
        for coeff in coeffs[:-1].flip(0):
            y = y * x + coeff.to(dtype=x.dtype, device=x.device)
        return y

    def forward(self, x):
        compute_x = x.float()
        scaled_x = compute_x / self.input_scale
        sign_approximation = scaled_x
        for index in range(1, len(self._COMPONENT_COEFFS) + 1):
            coefficients = getattr(self, "p{}_coeffs".format(index)).float()
            sign_approximation = self._polyval(
                sign_approximation, coefficients)
        out = self.input_scale * 0.5 * (
            scaled_x + scaled_x * sign_approximation)
        return out.to(dtype=x.dtype)


class PreciseReLUAlpha7(_PreciseReLU):
    """PreciseReLU with alpha=7 from Appendix A of the reference paper.

    The normalized approximation target is ReLU on ``[-1, 1]`` with error
    at most ``2**-7``. ``input_scale`` applies the paper's transformation
    ``B * r_7(x / B)`` to target ``[-B, B]`` (error at most ``B * 2**-7``).
    """

    alpha = 7
    component_degrees = (7, 7)
    multiplicative_depth = 7
    non_scalar_multiplications = 9
    _P1_COEFFS = (
        3.60471572275560e-36, 7.30445164958251,
        -5.05471704202722e-35, -3.46825871108659e1,
        1.16564665409095e-34, 5.98596518298826e1,
        -6.54298492839531e-35, -3.18755225906466e1,
    )
    _P2_COEFFS = (
        -9.46491402344260e-49, 2.40085652217597,
        6.41744632725342e-48, -2.63125454261783,
        -7.25338564676814e-48, 1.54912674773593,
        2.06916466421812e-48, -3.31172956504304e-1,
    )
    _COMPONENT_COEFFS = (_P1_COEFFS, _P2_COEFFS)


class PreciseReLUAlpha10(_PreciseReLU):
    """PreciseReLU with alpha=10 from Appendix A of the reference paper."""

    alpha = 10
    component_degrees = (7, 7, 13)
    multiplicative_depth = 11
    non_scalar_multiplications = 16
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
    _COMPONENT_COEFFS = (_P1_COEFFS, _P2_COEFFS, _P3_COEFFS)


_PRECISE_RELU_BY_ALPHA = {
    7: PreciseReLUAlpha7,
    10: PreciseReLUAlpha10,
}


class PReLU_Approx(torch.nn.Module):
    def __init__(self, slope, input_scale=1.0, polynomial_degree=4,
                 precise_alpha=None):
        super().__init__()
        slope = slope.detach().clone().float().reshape(-1)
        if precise_alpha is None:
            self.relu = ChebyReLU(
                input_scale=input_scale, degree=polynomial_degree)
        else:
            try:
                precise_relu = _PRECISE_RELU_BY_ALPHA[int(precise_alpha)]
            except (KeyError, TypeError, ValueError):
                raise ValueError('precise_alpha must be 7, 10, or None')
            self.relu = precise_relu(input_scale=input_scale)
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


def replace_resnet_activations_with_poly(
        module, input_scale=1.0, polynomial_degree=4, precise_alpha=None):
    replaced = 0
    for name, child in module.named_children():
        if isinstance(child, torch.nn.PReLU):
            replacement = PReLU_Approx(
                child.weight, input_scale=input_scale,
                polynomial_degree=polynomial_degree,
                precise_alpha=precise_alpha).to(
                    device=child.weight.device, dtype=child.weight.dtype)
            setattr(module, name, replacement)
            replaced += 1
        else:
            replaced += replace_resnet_activations_with_poly(
                child, input_scale=input_scale,
                polynomial_degree=polynomial_degree,
                precise_alpha=precise_alpha)
    return replaced


def _prelu_modules(module):
    return [
        (name, child)
        for name, child in module.named_modules()
        if name and isinstance(child, torch.nn.PReLU)
    ]


def _replace_named_child(module, qualified_name, child):
    parent_name, _, local_name = qualified_name.rpartition('.')
    parent = module.get_submodule(parent_name) if parent_name else module
    setattr(parent, local_name, child)


def replace_resnet_activations_with_poly_scales(
        module, input_scales, polynomial_degree=4, precise_alpha=None):
    """Replace every PReLU using a fixed public scale keyed by module name."""
    prelus = _prelu_modules(module)
    expected_names = {name for name, _ in prelus}
    supplied_names = set(input_scales)
    missing = sorted(expected_names - supplied_names)
    unknown = sorted(supplied_names - expected_names)
    if missing or unknown:
        raise ValueError(
            'Per-layer input scales do not match the model PReLUs; '
            'missing={}, unknown={}'.format(missing, unknown))

    for name, prelu in prelus:
        scale = float(input_scales[name])
        if not torch.isfinite(torch.tensor(scale)) or scale <= 0.0:
            raise ValueError(
                'Input scale for {} must be finite and positive'.format(name))
        replacement = PReLU_Approx(
            prelu.weight, scale, polynomial_degree=polynomial_degree,
            precise_alpha=precise_alpha).to(
            device=prelu.weight.device, dtype=prelu.weight.dtype)
        _replace_named_child(module, name, replacement)
    return len(prelus)


def calibrate_resnet_activations_with_poly(
        module, calibration_inputs, scale_margin=2.0, min_input_scale=1e-3,
        polynomial_degree=4, precise_alpha=None):
    """Sequentially measure and replace PReLUs with fixed-scale polynomials.

    The input of each activation is measured on the *partially converted*
    graph, after all earlier PReLUs have already been replaced.  Calibration
    is plaintext-only; the resulting ``input_scale`` buffers are fixed public
    constants during inference.
    """
    scale_margin = float(scale_margin)
    min_input_scale = float(min_input_scale)
    if scale_margin <= 1.0:
        raise ValueError('scale_margin must be greater than 1')
    if min_input_scale <= 0.0:
        raise ValueError('min_input_scale must be positive')
    if (precise_alpha is None
            and polynomial_degree not in ChebyReLU._NORMALIZED_POWER_COEFFS):
        raise ValueError('polynomial_degree must be 4, 8, or 16')
    if precise_alpha is not None and precise_alpha not in _PRECISE_RELU_BY_ALPHA:
        raise ValueError('precise_alpha must be 7, 10, or None')
    if not torch.is_tensor(calibration_inputs) or calibration_inputs.numel() == 0:
        raise ValueError('calibration_inputs must be a non-empty tensor')
    if not torch.isfinite(calibration_inputs).all():
        raise FloatingPointError('Calibration inputs contain non-finite values')

    prelu_names = [name for name, _ in _prelu_modules(module)]
    diagnostics = []
    was_training = module.training
    module.eval()
    try:
        with torch.no_grad():
            for name in prelu_names:
                prelu = module.get_submodule(name)
                observed = {}

                def capture_input(_child, inputs):
                    values = inputs[0]
                    if not torch.isfinite(values).all():
                        raise FloatingPointError(
                            'Non-finite calibration input reached {}'.format(name))
                    observed['absmax'] = float(values.detach().abs().max().item())

                handle = prelu.register_forward_pre_hook(capture_input)
                try:
                    partial_output = module(calibration_inputs)
                finally:
                    handle.remove()
                if not torch.isfinite(partial_output).all():
                    raise FloatingPointError(
                        'Partially converted model became non-finite before '
                        '{} could be calibrated'.format(name))
                if 'absmax' not in observed:
                    raise RuntimeError(
                        'PReLU {} was not executed during calibration'.format(name))

                input_absmax = observed['absmax']
                input_scale = max(input_absmax * scale_margin, min_input_scale)
                replacement = PReLU_Approx(
                    prelu.weight, input_scale,
                    polynomial_degree=polynomial_degree,
                    precise_alpha=precise_alpha).to(
                    device=prelu.weight.device, dtype=prelu.weight.dtype)
                _replace_named_child(module, name, replacement)
                diagnostics.append({
                    'module': name,
                    'input_absmax': input_absmax,
                    'input_scale': input_scale,
                })

            final_output = module(calibration_inputs)
            if not torch.isfinite(final_output).all():
                raise FloatingPointError(
                    'Fully converted model is non-finite on calibration inputs')
    finally:
        module.train(was_training)
    return diagnostics


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

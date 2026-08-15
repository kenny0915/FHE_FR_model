"""IResNet with sequentially calibrated, rescaled PReLU polynomials.

Every PReLU location owns an independent approximation interval ``[-S, S]``.
For the trained channel-wise PReLU slope ``a`` the normalized target is

    PReLU_a(z) = (1 + a) / 2 * z + (1 - a) / 2 * |z|,  z in [-1, 1].

The student is evaluated as ``S * q(x / S)`` and is constrained to agree
with PReLU at both interval endpoints.  A degree-2 student is

    student(x) = linear*x + even*x^2/S + beta2*(1 - z^2),

and the optional degree-3 term ``theta3*x*(1-z^2)`` also vanishes at the
endpoints.  ``beta2`` is an original-domain offset, so its gradient does not
grow with the interval scale.  Thus coefficient learning cannot break the
endpoint constraints.  Once fully converted, the activation folds to plaintext
channel-wise coefficients for ``c0 + c1*x + c2*x^2 (+ c3*x^3)``.

The scale is a public plaintext constant.  Degree 2 needs one sequential
ciphertext-ciphertext multiplication; degree 3 needs two.
"""

import torch
from torch import nn

from .iresnet_no_relu import IBasicBlock, IResNet as _ProgressiveIResNet

__all__ = [
    "FoldedLayerwisePolynomial",
    "LayerwisePolynomialActivation",
    "IResNet",
    "iresnet18",
    "iresnet34",
    "iresnet50",
    "iresnet100",
    "iresnet200",
]

_STAGE_NAMES = ("stem", "layer1", "layer2", "layer3", "layer4")


class FoldedLayerwisePolynomial(nn.Module):
    """Inference-only channel-wise polynomial in the original input domain."""

    def __init__(self, coefficients):
        super().__init__()
        if len(coefficients) not in (3, 4):
            raise ValueError("Only degree-2 and degree-3 polynomials are supported")
        self.degree = len(coefficients) - 1
        for index, coefficient in enumerate(coefficients):
            self.register_buffer(
                f"coefficient{index}", coefficient.detach().clone())

    def forward(self, x):
        compute_dtype = (
            torch.float32
            if x.dtype in (torch.float16, torch.bfloat16)
            else x.dtype
        )
        compute_x = x.to(dtype=compute_dtype)
        output = getattr(self, f"coefficient{self.degree}").to(
            device=x.device, dtype=compute_dtype)
        for index in range(self.degree - 1, -1, -1):
            coefficient = getattr(self, f"coefficient{index}").to(
                device=x.device, dtype=compute_dtype)
            output = output * compute_x + coefficient
        return output.to(dtype=x.dtype)


class LayerwisePolynomialActivation(nn.Module):
    """Progressively replace one trained PReLU by its own rescaled polynomial."""

    is_progressive_polynomial_activation = True
    is_layerwise_rescaled_polynomial = True
    exclude_from_weight_decay = True

    def __init__(self, channels, degree=2, initial_scale=1.0,
                 distill_eps=1e-4, stage_index=0, blend=0.0):
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        if degree not in (2, 3):
            raise ValueError("layerwise polynomial degree must be 2 or 3")
        if initial_scale <= 0:
            raise ValueError("initial_scale must be positive")
        if distill_eps <= 0:
            raise ValueError("distill_eps must be positive")

        self.degree = int(degree)
        self.stage_index = int(stage_index)
        self.prelu = nn.PReLU(channels)
        self.prelu.weight.requires_grad = False

        # beta2=0 initializes linear*x + even*x^2/S. This is exact at
        # x=-S, 0, S. Unlike the old normalized theta2, d(student)/d(beta2)
        # is bounded by one inside the calibrated interval for every S.
        self.beta2 = nn.Parameter(torch.zeros(channels, 1, 1))
        if self.degree == 3:
            self.theta3 = nn.Parameter(torch.zeros(channels, 1, 1))
        else:
            self.register_parameter("theta3", None)

        self.register_buffer(
            "input_scale", torch.tensor(float(initial_scale), dtype=torch.float32))
        self.register_buffer(
            "scale_calibrated", torch.tensor(False, dtype=torch.bool))
        self.register_buffer(
            "distill_eps", torch.tensor(float(distill_eps), dtype=torch.float32))
        self.register_buffer(
            "blend", torch.tensor(float(blend), dtype=torch.float32))
        self._blend = 0.0
        self._scale_is_calibrated = False
        self._last_range_penalty = None
        self._last_distillation_loss = None
        self._last_input_absmax = None
        self._last_outside_fraction = None
        self._loaded_legacy_theta2 = False
        self.set_blend(blend)

    def set_blend(self, blend):
        blend = float(blend)
        if not 0.0 <= blend <= 1.0:
            raise ValueError("blend must be in [0, 1]")
        if blend > 0.0 and not self._scale_is_calibrated:
            raise RuntimeError(
                "Calibrate this activation's input scale before conversion")
        self._blend = blend
        self.blend.fill_(blend)

    @torch.no_grad()
    def set_input_scale(self, scale):
        scale = float(scale)
        if not torch.isfinite(torch.tensor(scale)) or scale <= 0.0:
            raise ValueError("input scale must be finite and positive")
        if self._blend > 0.0:
            raise RuntimeError(
                "Cannot change an activation interval after conversion starts")
        self.input_scale.fill_(scale)
        self.scale_calibrated.fill_(True)
        self._scale_is_calibrated = True

    def range_penalty(self):
        return self._last_range_penalty

    def distillation_loss(self):
        return self._last_distillation_loss

    def range_stats(self):
        return {
            "absmax": self._last_input_absmax,
            "outside_fraction": self._last_outside_fraction,
            "blend": self.blend.detach(),
            "input_scale": self.input_scale.detach(),
            "scale_calibrated": self.scale_calibrated.detach(),
        }

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        old_key = prefix + "weight"
        teacher_key = prefix + "prelu.weight"
        if old_key in state_dict and teacher_key not in state_dict:
            state_dict[teacher_key] = state_dict.pop(old_key)

        beta2_key = prefix + "beta2"
        old_theta2_key = prefix + "theta2"
        has_beta2_state = beta2_key in state_dict
        has_legacy_theta2_state = old_theta2_key in state_dict
        has_student_state = has_beta2_state or has_legacy_theta2_state
        if has_legacy_theta2_state and not has_beta2_state:
            slope = state_dict.get(teacher_key, self.prelu.weight.detach())
            scale = state_dict.get(prefix + "input_scale", self.input_scale)
            even = 0.5 * (1.0 - slope.detach().float()).reshape(-1, 1, 1)
            state_dict[beta2_key] = (
                scale.detach().float() *
                (even + state_dict.pop(old_theta2_key).detach().float())
            )
            self._loaded_legacy_theta2 = True
        if not has_student_state:
            state_dict[beta2_key] = torch.zeros_like(self.beta2.detach())
            if self.degree == 3:
                state_dict[prefix + "theta3"] = torch.zeros_like(
                    self.beta2.detach())
            # A baseline checkpoint has not profiled an interval yet.
            state_dict[prefix + "scale_calibrated"] = torch.tensor(False)

        if not has_student_state:
            for local_key, value in self.state_dict().items():
                full_key = prefix + local_key
                if full_key not in state_dict:
                    state_dict[full_key] = value.detach()

        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)
        self._blend = float(self.blend.item())
        self._scale_is_calibrated = bool(self.scale_calibrated.item())

    def _student(self, x, detach_input=False):
        compute_dtype = (
            torch.float32
            if x.dtype in (torch.float16, torch.bfloat16)
            else x.dtype
        )
        compute_x = (x.detach() if detach_input else x).to(dtype=compute_dtype)
        scale = self.input_scale.to(device=x.device, dtype=compute_dtype)
        z = compute_x / scale
        slope = self.prelu.weight.detach().reshape(1, -1, 1, 1).to(
            device=x.device, dtype=compute_dtype)
        linear = 0.5 * (1.0 + slope)
        even = 0.5 * (1.0 - slope)
        square = compute_x.square()
        endpoint_basis = 1.0 - square / scale.square()
        beta2 = self.beta2.reshape(1, -1, 1, 1).to(
            device=x.device, dtype=compute_dtype)
        student = (
            linear * compute_x
            + even * square / scale
            + beta2 * endpoint_basis
        )
        if self.theta3 is not None:
            theta3 = self.theta3.reshape(1, -1, 1, 1).to(
                device=x.device, dtype=compute_dtype)
            student = student + theta3 * compute_x * endpoint_basis
        return student.to(dtype=x.dtype)

    def forward(self, x):
        calibrated = self._scale_is_calibrated
        if self.training:
            compute_x = (
                x.float()
                if x.dtype in (torch.float16, torch.bfloat16)
                else x
            )
            if calibrated:
                scale = self.input_scale.to(
                    device=x.device, dtype=compute_x.dtype)
                excess = torch.relu(compute_x.abs() - scale)
                self._last_range_penalty = (
                    excess.square().mean()
                    + 0.1 * excess.flatten(1).amax(dim=1).square().mean()
                )
                self._last_input_absmax = compute_x.detach().abs().amax()
                self._last_outside_fraction = (
                    (excess.detach() > 0).float().mean())
            else:
                zero = compute_x.sum() * 0.0
                self._last_range_penalty = zero
                self._last_input_absmax = None
                self._last_outside_fraction = None
        else:
            self._last_range_penalty = None
            self._last_distillation_loss = None

        blend = self._blend
        if not self.training and blend <= 0.0:
            return self.prelu(x)

        teacher = self.prelu(x)
        student = self._student(x)
        if self.training:
            if calibrated:
                local_student = self._student(x, detach_input=True)
                target = teacher.detach().float()
                denominator = target.square().mean().detach()
                denominator = denominator + self.distill_eps.to(
                    device=x.device, dtype=denominator.dtype)
                self._last_distillation_loss = (
                    local_student.float() - target).square().mean() / denominator
            else:
                # Keep every student parameter in DDP's stationary graph before
                # its scale is profiled, without applying an invalid target loss.
                self._last_distillation_loss = student.sum() * 0.0

        if blend <= 0.0:
            return teacher + student * 0.0
        if blend >= 1.0:
            return student + teacher * 0.0 if self.training else student
        return (1.0 - blend) * teacher + blend * student

    @torch.no_grad()
    def folded_coefficients(self):
        scale = self.input_scale.to(
            device=self.beta2.device, dtype=self.beta2.dtype)
        slope = self.prelu.weight.detach().reshape(-1, 1, 1).to(
            device=self.beta2.device, dtype=self.beta2.dtype)
        linear = 0.5 * (1.0 + slope)
        even = 0.5 * (1.0 - slope)
        coefficient0 = self.beta2
        coefficient1 = linear
        coefficient2 = even / scale - self.beta2 / scale.square()
        coefficients = [coefficient0, coefficient1, coefficient2]
        if self.theta3 is not None:
            coefficients[1] = coefficient1 + self.theta3
            coefficients.append(-self.theta3 / scale.square())
        return tuple(coefficients)

    @torch.no_grad()
    def folded(self):
        if self._blend < 1.0:
            raise RuntimeError(
                "Only a fully converted layerwise polynomial can be folded")
        if not self._scale_is_calibrated:
            raise RuntimeError("Cannot fold an activation with no calibrated interval")
        return FoldedLayerwisePolynomial(self.folded_coefficients())


class IResNet(_ProgressiveIResNet):
    """IResNet whose PReLUs are converted and calibrated in forward order."""

    def __init__(self, *args, layerwise_poly_degree=2,
                 layerwise_poly_initial_scale=1.0,
                 layerwise_poly_distill_eps=1e-4,
                 layerwise_poly_progress=0.0, **kwargs):
        object.__setattr__(self, "layerwise_poly_degree", int(layerwise_poly_degree))
        object.__setattr__(
            self, "layerwise_poly_initial_scale",
            float(layerwise_poly_initial_scale))
        object.__setattr__(
            self, "layerwise_poly_distill_eps",
            float(layerwise_poly_distill_eps))
        super().__init__(
            *args, herpn_progress=float(layerwise_poly_progress), **kwargs)

    def _make_activation(self, channels, stage_name):
        return LayerwisePolynomialActivation(
            channels=channels,
            degree=self.layerwise_poly_degree,
            initial_scale=self.layerwise_poly_initial_scale,
            distill_eps=self.layerwise_poly_distill_eps,
            stage_index=_STAGE_NAMES.index(stage_name),
            blend=0.0,
        )

    def progressive_activations(self):
        return [
            module for module in self.modules()
            if isinstance(module, LayerwisePolynomialActivation)
        ]

    def named_progressive_activations(self):
        return [
            (name, module) for name, module in self.named_modules()
            if isinstance(module, LayerwisePolynomialActivation)
        ]

    def set_herpn_progress(self, progress):
        progress = min(max(float(progress), 0.0), float(len(_STAGE_NAMES)))
        self.herpn_progress.fill_(progress)
        for activation in self.progressive_activations():
            activation.set_blend(
                min(max(progress - activation.stage_index, 0.0), 1.0))

    def set_herpn_blends(self, blends):
        activations = dict(self.named_progressive_activations())
        unknown = sorted(set(blends).difference(activations))
        if unknown:
            raise ValueError(
                "Unknown layerwise polynomial activation names: {}".format(unknown))
        for name, activation in activations.items():
            activation.set_blend(float(blends.get(name, 0.0)))
        converted_fraction = sum(
            activation._blend for activation in activations.values()
        ) / len(activations)
        self.herpn_progress.fill_(converted_fraction * len(_STAGE_NAMES))

    def layerwise_poly_activation_names(self):
        return [name for name, _ in self.named_progressive_activations()]

    def uncalibrated_layerwise_poly_names(self):
        return [
            name for name, activation in self.named_progressive_activations()
            if not activation._scale_is_calibrated
        ]

    def legacy_layerwise_poly_parameters(self):
        """Parameters whose legacy theta2 optimizer state must be discarded."""
        return [
            activation.beta2
            for activation in self.progressive_activations()
            if activation._loaded_legacy_theta2
        ]

    @torch.no_grad()
    def set_layerwise_poly_input_scale(self, name, scale):
        activations = dict(self.named_progressive_activations())
        if name not in activations:
            raise ValueError(f"Unknown layerwise polynomial activation: {name}")
        activations[name].set_input_scale(scale)

    def herpn_range_stats(self):
        return {
            name: activation.range_stats()
            for name, activation in self.named_progressive_activations()
        }

    def herpn_range_penalty(self):
        penalties = [
            activation.range_penalty()
            for activation in self.progressive_activations()
            if activation._scale_is_calibrated
            and activation.range_penalty() is not None
        ]
        if not penalties:
            return next(self.parameters()).new_zeros(())
        return torch.stack(penalties).mean()

    def herpn_distillation_loss(self):
        losses = [
            activation.distillation_loss()
            for activation in self.progressive_activations()
            if activation._scale_is_calibrated
            and activation.distillation_loss() is not None
        ]
        if not losses:
            return next(self.parameters()).new_zeros(())
        return torch.stack(losses).mean()

    def begin_batchnorm_recalibration_after(self, activation_name, reset=True):
        """Refresh only BN modules downstream of an already measured prefix."""
        named_modules = list(self.named_modules())
        module_names = [name for name, _ in named_modules]
        if activation_name not in module_names:
            raise ValueError(
                f"Unknown activation for downstream BN refresh: {activation_name}")
        activation_index = module_names.index(activation_name)
        batchnorm_state = [
            (module, module.training, module.momentum)
            for _, module in named_modules[activation_index + 1:]
            if isinstance(module, nn.modules.batchnorm._BatchNorm)
        ]
        state = {
            "model_training": self.training,
            "batchnorm": batchnorm_state,
        }
        self.eval()
        for module, _, _ in batchnorm_state:
            if reset:
                module.reset_running_stats()
            module.momentum = None
            module.train()
        return state

    @torch.no_grad()
    def fold_layerwise_polynomials_for_inference(self):
        if self.training:
            raise RuntimeError("Call eval() before folding layerwise polynomials")
        if any(
                activation._blend < 1.0
                for activation in self.progressive_activations()):
            raise RuntimeError(
                "All layerwise polynomial activations must be fully converted")

        def replace(module):
            for name, child in list(module.named_children()):
                if isinstance(child, LayerwisePolynomialActivation):
                    setattr(module, name, child.folded())
                else:
                    replace(child)

        replace(self)
        return self

    # Keep the trainer/export protocol shared with the other polynomial R50s.
    fold_herpn_for_inference = fold_layerwise_polynomials_for_inference


def _iresnet(blocks, pretrained, **kwargs):
    model = IResNet(IBasicBlock, blocks, **kwargs)
    if pretrained:
        raise ValueError("No bundled pretrained layerwise-polynomial checkpoint")
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

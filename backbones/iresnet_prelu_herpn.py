"""IResNet with a PReLU-aware, degree-2 HerPN activation.

For a channel-wise PReLU slope ``a``,

    PReLU_a(x) = a*x + (1-a)*ReLU(x).

This module replaces only the ReLU term by the basis-normalized HerPN from
AESPA:

    student_a(x) = a*x + (1-a)*HerPN_ReLU(x).

The optional layerwise-scaled form calibrates a public ``S_i`` at every
activation and evaluates ``S_i*HerPN_ReLU(x/S_i)``. PReLU's positive
homogeneity keeps the teacher target unchanged while the Hermite square sees
the normalized interval ``[-1, 1]``.

After calibration, ``HerPN_ReLU(x)`` folds to ``A*x^2+B*x+C``.  Therefore
the complete student folds exactly to

    (1-a)*A*x^2 + (a+(1-a)*B)*x + (1-a)*C.

The encrypted activation remains degree two and requires one sequential
ciphertext-ciphertext multiplication for ``x^2``.  The channel slope and
folded coefficients are plaintext constants during inference.
"""

import math

import torch
from torch import nn

from .iresnet_no_relu import (
    FoldedHerPN,
    HerPN,
    IBasicBlock,
    IResNet as _HerPNIResNet,
)

__all__ = [
    "PReLUHerPNActivation",
    "IResNet",
    "iresnet18",
    "iresnet34",
    "iresnet50",
    "iresnet100",
    "iresnet200",
]

_STAGE_NAMES = ("stem", "layer1", "layer2", "layer3", "layer4")


class _PReLUHerPN(HerPN):
    """HerPN variant exposing its basis for local activation distillation."""

    exclude_from_weight_decay = True

    def __init__(self, channels, eps):
        super().__init__(channels, eps=eps)
        # At blend=0 the teacher supplies the actual network output. Starting
        # the student at a*x (zero ReLU branch) gives an O(1) relative error;
        # HerPN's default unit scale can be 15x larger than deep PReLU targets.
        nn.init.zeros_(self.weight)
        nn.init.zeros_(self.bias)

    def forward_with_basis(self, x):
        compute_dtype = (
            torch.float32
            if x.dtype in (torch.float16, torch.bfloat16)
            else x.dtype
        )
        compute_x = x.to(dtype=compute_dtype)
        x0 = self.bn0(torch.ones_like(compute_x))
        x1 = self.bn1(compute_x)
        x2 = self.bn2(
            (compute_x.square() - 1.0) / math.sqrt(2.0))
        basis = (
            x0 / math.sqrt(2.0 * math.pi)
            + x1 / 2.0
            + x2 / math.sqrt(4.0 * math.pi)
        )
        output = self.weight.to(dtype=compute_dtype) * basis
        output = output + self.bias.to(dtype=compute_dtype)
        return output.to(dtype=x.dtype), basis

    def forward(self, x):
        output, _ = self.forward_with_basis(x)
        return output


class PReLUHerPNActivation(nn.Module):
    """Progressively blend a frozen PReLU teacher into its HerPN student."""

    is_progressive_polynomial_activation = True
    is_layerwise_rescaled_polynomial = True

    def __init__(self, channels, range_limit=6.0, bn_eps=1e-4,
                 distill_eps=1e-4, stage_index=0, blend=1.0,
                 layerwise_scale=False, initial_scale=1.0):
        super().__init__()
        if range_limit <= 0:
            raise ValueError("range_limit must be positive")
        if distill_eps <= 0:
            raise ValueError("distill_eps must be positive")
        if initial_scale <= 0:
            raise ValueError("initial_scale must be positive")

        self.prelu = nn.PReLU(channels)
        # The pretrained PReLU is the fixed teacher and supplies the
        # channel-wise plaintext slope in the polynomial student.
        self.prelu.weight.requires_grad = False
        self.herpn = _PReLUHerPN(channels, eps=bn_eps)
        self.stage_index = int(stage_index)
        self.register_buffer(
            "blend", torch.tensor(float(blend), dtype=torch.float32))
        self.register_buffer(
            "range_limit",
            torch.tensor(float(range_limit), dtype=torch.float32))
        self.register_buffer(
            "distill_eps",
            torch.tensor(float(distill_eps), dtype=torch.float32))
        self.register_buffer(
            "input_scale",
            torch.tensor(float(initial_scale), dtype=torch.float32))
        self.register_buffer(
            "scale_calibrated", torch.tensor(False, dtype=torch.bool))
        self.register_buffer(
            "layerwise_scale_enabled",
            torch.tensor(bool(layerwise_scale), dtype=torch.bool))
        self._last_range_penalty = None
        self._last_distillation_loss = None
        self._last_input_absmax = None
        self._last_outside_fraction = None
        self._blend = 0.0
        self._scale_is_calibrated = False
        self.set_blend(blend)

    def set_blend(self, blend):
        blend = float(blend)
        if not 0.0 <= blend <= 1.0:
            raise ValueError("blend must be in [0, 1]")
        if (blend > 0.0 and bool(self.layerwise_scale_enabled.item())
                and not self._scale_is_calibrated):
            raise RuntimeError(
                "Calibrate this HerPN activation's input scale before conversion")
        self._blend = blend
        self.blend.fill_(blend)

    @torch.no_grad()
    def set_input_scale(self, scale):
        scale = float(scale)
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError("input scale must be finite and positive")
        if not bool(self.layerwise_scale_enabled.item()):
            raise RuntimeError("Layerwise scaling is disabled for this activation")
        if self._blend > 0.0:
            raise RuntimeError(
                "Cannot change a HerPN interval after conversion starts")
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
        # Convert an ordinary IResNet key such as layer1.0.prelu.weight into
        # the progressive wrapper's frozen teacher key.
        old_key = prefix + "weight"
        teacher_key = prefix + "prelu.weight"
        if old_key in state_dict and teacher_key not in state_dict:
            state_dict[teacher_key] = state_dict.pop(old_key)

        herpn_prefix = prefix + "herpn."
        has_herpn_state = any(
            key.startswith(herpn_prefix) for key in state_dict)
        scaled_key = prefix + "layerwise_scale_enabled"
        scale_key = prefix + "input_scale"
        calibrated_key = prefix + "scale_calibrated"
        if scaled_key not in state_dict:
            state_dict[scaled_key] = self.layerwise_scale_enabled.detach()
        if scale_key not in state_dict:
            state_dict[scale_key] = self.input_scale.detach()
        if calibrated_key not in state_dict:
            # Old HerPN checkpoints represent the unscaled S=1 graph. Mark
            # that interval usable only when loading them in legacy mode.
            legacy_calibrated = (
                has_herpn_state
                and not bool(self.layerwise_scale_enabled.item()))
            state_dict[calibrated_key] = torch.tensor(legacy_calibrated)
        if not has_herpn_state:
            # A baseline PReLU checkpoint has no student or progressive state.
            # Fill all new state so strict loading still rejects a partially
            # written PReLU-HerPN checkpoint.
            for local_key, value in self.state_dict().items():
                full_key = prefix + local_key
                if full_key not in state_dict:
                    state_dict[full_key] = value.detach()

        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)
        self._blend = float(self.blend.item())
        self._scale_is_calibrated = bool(self.scale_calibrated.item())

    def _normalized_input_and_scale(self, x):
        if not bool(self.layerwise_scale_enabled.item()):
            return x, None
        scale = self.input_scale.to(device=x.device, dtype=x.dtype)
        return x / scale, scale

    def _student(self, x):
        normalized_x, scale = self._normalized_input_and_scale(x)
        relu_student = self.herpn(normalized_x)
        if scale is not None:
            relu_student = scale * relu_student
        slope = self.prelu.weight.reshape(1, -1, 1, 1).to(
            device=x.device, dtype=x.dtype)
        return slope * x + (1.0 - slope) * relu_student

    def _student_and_local_student(self, x):
        """Return the task student and a value-equal input-detached student.

        Both outputs share one HerPN basis evaluation. The task student keeps
        the normal derivative through ``x``. The local copy detaches only the
        basis and input, so activation distillation updates HerPN's scale and
        bias without modifying earlier pretrained backbone layers.
        """
        normalized_x, scale = self._normalized_input_and_scale(x)
        relu_student, basis = self.herpn.forward_with_basis(normalized_x)
        if scale is not None:
            relu_student = scale * relu_student
        slope = self.prelu.weight.reshape(1, -1, 1, 1).to(
            device=x.device, dtype=x.dtype)
        student = slope * x + (1.0 - slope) * relu_student

        compute_dtype = basis.dtype
        local_relu_student = (
            self.herpn.weight.to(dtype=compute_dtype) * basis.detach()
            + self.herpn.bias.to(dtype=compute_dtype)
        ).to(dtype=x.dtype)
        if scale is not None:
            local_relu_student = scale * local_relu_student
        local_student = (
            slope.detach() * x.detach()
            + (1.0 - slope.detach()) * local_relu_student
        )
        return student, local_student

    def forward(self, x):
        if self.training:
            compute_x = (
                x.float()
                if x.dtype in (torch.float16, torch.bfloat16)
                else x
            )
            scaled = bool(self.layerwise_scale_enabled.item())
            calibrated = self._scale_is_calibrated
            limit = (
                self.input_scale if scaled else self.range_limit
            ).to(device=x.device, dtype=compute_x.dtype)
            excess = torch.relu(compute_x.abs() - limit)
            if not scaled or calibrated:
                # Layerwise intervals can differ by many orders of magnitude.
                # Penalize relative escape so changing the public scale S does
                # not multiply this auxiliary objective and its gradient by
                # S^2.  Legacy unscaled activations retain their raw penalty.
                penalty_excess = excess / limit if scaled else excess
                self._last_range_penalty = (
                    penalty_excess.square().mean()
                    + 0.1
                    * penalty_excess.flatten(1).amax(dim=1).square().mean()
                )
                self._last_input_absmax = compute_x.detach().abs().amax()
                self._last_outside_fraction = (
                    (penalty_excess.detach() > 0).float().mean())
            else:
                self._last_range_penalty = compute_x.sum() * 0.0
                self._last_input_absmax = None
                self._last_outside_fraction = None
        else:
            self._last_range_penalty = None
            self._last_distillation_loss = None

        blend = self._blend
        if not self.training and blend <= 0.0:
            return self.prelu(x)

        teacher = self.prelu(x)
        if self.training:
            student, local_student = self._student_and_local_student(x)
            # Normalize by teacher energy so shrinking activation scales cannot
            # make the teacher constraint disappear.  Keep this active after
            # full conversion to prevent long-run polynomial drift. Distill
            # only HerPN's affine coefficients; task loss remains responsible
            # for adapting preceding layers to the polynomial graph.
            target = teacher.detach().float()
            denominator = target.square().mean().detach()
            denominator = denominator + self.distill_eps.to(
                device=x.device, dtype=denominator.dtype)
            if (not bool(self.layerwise_scale_enabled.item())
                    or self._scale_is_calibrated):
                self._last_distillation_loss = (
                    local_student.float() - target).square().mean() / denominator
            else:
                self._last_distillation_loss = local_student.sum() * 0.0
        else:
            student = self._student(x)

        if blend <= 0.0:
            return teacher + student * 0.0
        if blend >= 1.0:
            return (
                student + teacher * 0.0
                if self.training else student
            )
        return (1.0 - blend) * teacher + blend * student

    @torch.no_grad()
    def folded(self):
        if self._blend < 1.0:
            raise RuntimeError(
                "Only a fully converted PReLU-HerPN activation can be folded")
        coefficient2, coefficient1, coefficient0 = (
            self.herpn.folded_coefficients())
        if bool(self.layerwise_scale_enabled.item()):
            if not self._scale_is_calibrated:
                raise RuntimeError(
                    "Cannot fold a HerPN activation with no calibrated interval")
            scale = self.input_scale.to(
                device=coefficient2.device, dtype=coefficient2.dtype)
            coefficient2 = coefficient2 / scale
            coefficient0 = coefficient0 * scale
        slope = self.prelu.weight.detach().reshape(-1, 1, 1).to(
            device=coefficient1.device, dtype=coefficient1.dtype)
        residual = 1.0 - slope
        return FoldedHerPN(
            residual * coefficient2,
            slope + residual * coefficient1,
            residual * coefficient0,
        )


class IResNet(_HerPNIResNet):
    """IResNet topology with channel-wise PReLU-aware HerPN students."""

    def __init__(self, *args, prelu_herpn_distill_eps=1e-4,
                 prelu_herpn_layerwise_scale=False,
                 prelu_herpn_initial_scale=1.0, **kwargs):
        object.__setattr__(
            self, "prelu_herpn_distill_eps",
            float(prelu_herpn_distill_eps))
        object.__setattr__(
            self, "layerwise_input_scale_enabled",
            bool(prelu_herpn_layerwise_scale))
        object.__setattr__(
            self, "prelu_herpn_initial_scale",
            float(prelu_herpn_initial_scale))
        super().__init__(*args, **kwargs)

    def _make_activation(self, channels, stage_name):
        return PReLUHerPNActivation(
            channels=channels,
            range_limit=self.herpn_range_limit,
            bn_eps=self.herpn_bn_eps,
            distill_eps=self.prelu_herpn_distill_eps,
            stage_index=_STAGE_NAMES.index(stage_name),
            blend=0.0,
            layerwise_scale=self.layerwise_input_scale_enabled,
            initial_scale=self.prelu_herpn_initial_scale,
        )

    def progressive_activations(self):
        return [
            module for module in self.modules()
            if isinstance(module, PReLUHerPNActivation)
        ]

    def named_progressive_activations(self):
        return [
            (name, module) for name, module in self.named_modules()
            if isinstance(module, PReLUHerPNActivation)
        ]

    def layerwise_poly_activation_names(self):
        return [name for name, _ in self.named_progressive_activations()]

    def layerwise_poly_parameters(self, activation_names=None):
        """Return trainable HerPN coefficients for selected activations.

        Staged range conditioning keeps ordinary backbone gradients while
        suppressing polynomial updates outside the pending group.  The frozen
        PReLU slopes are teachers, and the internal Hermite BatchNorms are
        non-affine, so only the HerPN output weight and bias belong here.
        """
        selected = (
            None if activation_names is None else set(activation_names)
        )
        named_activations = self.named_progressive_activations()
        known = {name for name, _ in named_activations}
        if selected is not None:
            unknown = sorted(selected.difference(known))
            if unknown:
                raise ValueError(
                    f"Unknown PReLU-HerPN activations: {unknown}")

        parameters = []
        for name, activation in named_activations:
            if selected is None or name in selected:
                parameters.extend((
                    activation.herpn.weight,
                    activation.herpn.bias,
                ))
        return parameters

    def uncalibrated_layerwise_poly_names(self):
        return [
            name for name, activation in self.named_progressive_activations()
            if not activation._scale_is_calibrated
        ]

    @torch.no_grad()
    def set_layerwise_poly_input_scale(self, name, scale):
        activations = dict(self.named_progressive_activations())
        if name not in activations:
            raise ValueError(f"Unknown PReLU-HerPN activation: {name}")
        activations[name].set_input_scale(scale)

    def set_herpn_blends(self, blends):
        activations = {
            name: module for name, module in self.named_modules()
            if isinstance(module, PReLUHerPNActivation)
        }
        unknown = sorted(set(blends).difference(activations))
        if unknown:
            raise ValueError(
                "Unknown PReLU-HerPN activation names: {}".format(unknown))
        for name, activation in activations.items():
            activation.set_blend(float(blends.get(name, 0.0)))
        converted_fraction = sum(
            activation._blend for activation in activations.values()
        ) / len(activations)
        self.herpn_progress.fill_(
            converted_fraction * len(_STAGE_NAMES))

    def herpn_range_stats(self):
        return {
            name: module.range_stats()
            for name, module in self.named_modules()
            if isinstance(module, PReLUHerPNActivation)
        }

    def herpn_range_penalty(self, activation_names=None):
        selected = None if activation_names is None else set(activation_names)
        penalties = [
            activation.range_penalty()
            for name, activation in self.named_progressive_activations()
            if (selected is None or name in selected)
            and (not self.layerwise_input_scale_enabled
                 or activation._scale_is_calibrated)
            and activation.range_penalty() is not None
        ]
        if not penalties:
            return next(self.parameters()).new_zeros(())
        return torch.stack(penalties).mean()

    def herpn_distillation_loss(self, activation_names=None):
        selected = None if activation_names is None else set(activation_names)
        losses = [
            activation.distillation_loss()
            for name, activation in self.named_progressive_activations()
            if (selected is None or name in selected)
            and (not self.layerwise_input_scale_enabled
                 or activation._scale_is_calibrated)
            and activation.distillation_loss() is not None
        ]
        if not losses:
            return next(self.parameters()).new_zeros(())
        return torch.stack(losses).mean()

    def begin_batchnorm_recalibration_after(self, activation_name, reset=True):
        """Refresh only BatchNorms downstream of a converted activation."""
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
        state = {"model_training": self.training, "batchnorm": batchnorm_state}
        self.eval()
        for module, _, _ in batchnorm_state:
            if reset:
                module.reset_running_stats()
            module.momentum = None
            module.train()
        return state

    @torch.no_grad()
    def fold_herpn_for_inference(self):
        if self.training:
            raise RuntimeError("Call eval() before folding PReLU-HerPN")
        activations = self.progressive_activations()
        if any(activation._blend < 1.0 for activation in activations):
            raise RuntimeError(
                "All PReLU-HerPN activations must be fully converted "
                "before folding")

        def replace(module):
            for name, child in list(module.named_children()):
                if isinstance(child, PReLUHerPNActivation):
                    setattr(module, name, child.folded())
                else:
                    replace(child)

        replace(self)
        return self


def _iresnet(blocks, pretrained, **kwargs):
    model = IResNet(IBasicBlock, blocks, **kwargs)
    if pretrained:
        raise ValueError("No bundled pretrained PReLU-HerPN checkpoint")
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

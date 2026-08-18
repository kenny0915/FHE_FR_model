"""IResNet curricula from PreciseReLUAlpha10 to polynomial ReLU students.

Every original channel-wise PReLU is written as

    PReLU_a(x) = a*x + (1-a)*ReLU(x)

and only its ReLU term is replaced.  Training starts with the accurate
Alpha-10 composite approximation on ``[-S, S]`` and then smoothly transitions
through configured precise-alpha and/or independently fitted Chebyshev/minimax
students. The students are not coefficient truncations of Alpha-10: truncating
a jointly fitted power series is not a valid lower-degree approximation.

At a completed stage the encrypted activation is polynomial-only.  The PReLU
slope is a learned plaintext channel-wise coefficient.  The default final
degree 4 needs multiplicative depth 2 for the nonlinear power schedule.
"""

import math

import torch
from torch import nn

from .iresnet import IBasicBlock, IResNet as _IResNet
from .polynomial_relu import (
    ChebyReLU,
    PreciseReLUAlpha7,
    PreciseReLUAlpha10,
)

__all__ = [
    "ProgressivePrecisePReLU",
    "IResNet",
    "iresnet18",
    "iresnet34",
    "iresnet50",
    "iresnet100",
    "iresnet200",
]

_SUPPORTED_LOWER_DEGREES = (16, 8, 4)
_PRECISE_RELU_BY_ALPHA = {7: PreciseReLUAlpha7}


class _PolynomialRangePenaltyFunction(torch.autograd.Function):
    """Tail penalty that saves no intermediate feature maps for backward."""

    @staticmethod
    def forward(ctx, x, input_scale):
        compute_x = x.float()
        scale = input_scale.to(device=x.device, dtype=compute_x.dtype)
        excess = torch.relu(compute_x.abs() - scale)
        flat_excess = excess.flatten(1)
        penalty = (
            excess.square().mean()
            + 0.1 * flat_excess.amax(dim=1).square().mean()
        )
        ctx.save_for_backward(x, input_scale)
        return penalty

    @staticmethod
    def backward(ctx, grad_output):
        x, input_scale = ctx.saved_tensors
        compute_x = x.float()
        scale = input_scale.to(device=x.device, dtype=compute_x.dtype)
        excess = torch.relu(compute_x.abs() - scale)
        input_gradient = (
            2.0 * excess * compute_x.sign() / max(excess.numel(), 1))

        flat_excess = excess.flatten(1)
        sample_max, sample_argmax = flat_excess.max(dim=1)
        max_gradient = torch.zeros_like(flat_excess)
        max_gradient.scatter_(
            1,
            sample_argmax.unsqueeze(1),
            (0.2 * sample_max / max(flat_excess.shape[0], 1)).unsqueeze(1),
        )
        input_gradient = input_gradient + (
            max_gradient.reshape_as(compute_x) * compute_x.sign())
        input_gradient = input_gradient * grad_output.to(
            dtype=input_gradient.dtype)
        return input_gradient.to(dtype=x.dtype), None


class ProgressivePrecisePReLU(nn.Module):
    """PReLU using an Alpha10-to-polynomial-student curriculum."""

    is_progressive_precise_relu = True

    def __init__(self, channels, input_scale=8.0, lower_degrees=(16, 8, 4),
                 target_alphas=(), progress=0.0, initial_slope=0.25,
                 backward_mode="exact"):
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        if input_scale <= 0:
            raise ValueError("input_scale must be positive")
        lower_degrees = tuple(int(degree) for degree in lower_degrees)
        target_alphas = tuple(int(alpha) for alpha in target_alphas)
        if not target_alphas and not lower_degrees:
            raise ValueError(
                "target_alphas and lower_degrees must not both be empty")
        if any(alpha not in _PRECISE_RELU_BY_ALPHA
               for alpha in target_alphas):
            raise ValueError("target_alphas may contain only 7")
        if len(set(target_alphas)) != len(target_alphas):
            raise ValueError("target_alphas must not contain duplicates")
        if any(degree not in _SUPPORTED_LOWER_DEGREES
               for degree in lower_degrees):
            raise ValueError("lower_degrees may contain only 16, 8, and 4")
        if any(right >= left for left, right in zip(
                lower_degrees, lower_degrees[1:])):
            raise ValueError("lower_degrees must be strictly decreasing")

        self.prelu = nn.PReLU(channels, init=float(initial_slope))
        self.alpha10 = PreciseReLUAlpha10(
            input_scale=input_scale, backward_mode=backward_mode)
        precise_students = [
            _PRECISE_RELU_BY_ALPHA[alpha](
                input_scale=input_scale,
                backward_mode=backward_mode,
            )
            for alpha in target_alphas
        ]
        degree_students = [
            ChebyReLU(
                input_scale=input_scale,
                degree=degree,
                backward_mode=backward_mode,
            )
            for degree in lower_degrees
        ]
        self.students = nn.ModuleList(precise_students + degree_students)
        self.backward_mode = str(backward_mode)
        self.target_alphas = target_alphas
        self.lower_degrees = lower_degrees
        self.register_buffer(
            "progress", torch.tensor(float(progress), dtype=torch.float32))
        self._progress = 0.0
        self._last_range_penalty = None
        self._last_input_absmax = None
        self._last_outside_fraction = None
        self.set_degree_progress(progress)

    @property
    def transition_count(self):
        return len(self.students)

    def set_degree_progress(self, progress):
        progress = float(progress)
        if not math.isfinite(progress):
            raise ValueError("polynomial progress must be finite")
        progress = min(max(progress, 0.0), float(self.transition_count))
        self._progress = progress
        self.progress.fill_(progress)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # Ordinary IResNet checkpoints store ``prelu.weight`` at this module's
        # location. Move it into the wrapper and populate fixed polynomial
        # state so strict loading still detects partial curriculum checkpoints.
        old_key = prefix + "weight"
        new_key = prefix + "prelu.weight"
        baseline_checkpoint = old_key in state_dict and new_key not in state_dict
        if baseline_checkpoint:
            state_dict[new_key] = state_dict.pop(old_key)
            for local_key, value in self.state_dict().items():
                full_key = prefix + local_key
                if full_key not in state_dict:
                    state_dict[full_key] = value.detach()
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)
        self._progress = float(self.progress.item())

    def range_penalty(self):
        return self._last_range_penalty

    def range_stats(self):
        return {
            "absmax": self._last_input_absmax,
            "outside_fraction": self._last_outside_fraction,
            "progress": self.progress.detach(),
        }

    def _relu_stage(self, stage_index, x):
        if stage_index == 0:
            return self.alpha10(x)
        return self.students[stage_index - 1](x)

    def forward(self, x):
        if self.training:
            compute_x = (
                x.float()
                if x.dtype in (torch.float16, torch.bfloat16)
                else x
            )
            scale = self.alpha10.input_scale.to(
                device=x.device, dtype=compute_x.dtype)
            self._last_range_penalty = _PolynomialRangePenaltyFunction.apply(
                x, self.alpha10.input_scale)
            with torch.no_grad():
                excess = torch.relu(compute_x.abs() - scale)
                self._last_input_absmax = compute_x.abs().amax()
                self._last_outside_fraction = (
                    (excess > 0).float().mean())
        else:
            self._last_range_penalty = None

        lower_stage = min(
            int(math.floor(self._progress)), self.transition_count)
        blend = self._progress - lower_stage
        lower_relu = self._relu_stage(lower_stage, x)
        if blend > 0.0 and lower_stage < self.transition_count:
            upper_relu = self._relu_stage(lower_stage + 1, x)
            relu_out = (1.0 - blend) * lower_relu + blend * upper_relu
        else:
            relu_out = lower_relu

        slope = self.prelu.weight.to(dtype=x.dtype, device=x.device)
        slope = slope.reshape(1, slope.numel(), *([1] * (x.ndim - 2)))
        return slope * x + (1.0 - slope) * relu_out


class IResNet(_IResNet):
    """Ordinary IResNet topology with all 25 R50 PReLUs replaced."""

    def __init__(self, *args, precise_relu_input_scale=8.0,
                 precise_relu_target_alphas=(),
                 precise_relu_lower_degrees=(16, 8, 4),
                 precise_relu_progress=0.0,
                 precise_relu_backward_mode="exact", **kwargs):
        super().__init__(*args, **kwargs)
        self.precise_relu_input_scale = float(precise_relu_input_scale)
        self.precise_relu_target_alphas = tuple(
            int(alpha) for alpha in precise_relu_target_alphas)
        self.precise_relu_lower_degrees = tuple(
            int(degree) for degree in precise_relu_lower_degrees)
        self.precise_relu_backward_mode = str(
            precise_relu_backward_mode)
        self.register_buffer(
            "polynomial_progress",
            torch.tensor(float(precise_relu_progress), dtype=torch.float32),
            persistent=False)
        self._replace_all_prelus(self)
        self.set_polynomial_progress(precise_relu_progress)

    def _replace_all_prelus(self, module):
        for name, child in list(module.named_children()):
            if isinstance(child, nn.PReLU):
                replacement = ProgressivePrecisePReLU(
                    channels=child.num_parameters,
                    input_scale=self.precise_relu_input_scale,
                    target_alphas=self.precise_relu_target_alphas,
                    lower_degrees=self.precise_relu_lower_degrees,
                    progress=float(self.polynomial_progress.item()),
                    backward_mode=self.precise_relu_backward_mode,
                ).to(device=child.weight.device, dtype=child.weight.dtype)
                with torch.no_grad():
                    replacement.prelu.weight.copy_(child.weight)
                setattr(module, name, replacement)
            else:
                self._replace_all_prelus(child)

    def polynomial_activations(self):
        return [
            module for module in self.modules()
            if isinstance(module, ProgressivePrecisePReLU)
        ]

    def polynomial_transition_count(self):
        return (
            len(self.precise_relu_target_alphas)
            + len(self.precise_relu_lower_degrees)
        )

    def set_polynomial_progress(self, progress):
        progress = min(max(float(progress), 0.0),
                       float(self.polynomial_transition_count()))
        self.polynomial_progress.fill_(progress)
        for activation in self.polynomial_activations():
            activation.set_degree_progress(progress)

    def polynomial_stage_names(self):
        return (
            ("alpha10",)
            + tuple("alpha{}".format(alpha)
                    for alpha in self.precise_relu_target_alphas)
            + tuple(
                "degree{}".format(degree)
                for degree in self.precise_relu_lower_degrees)
        )

    def polynomial_range_penalty(self):
        penalties = [
            activation.range_penalty()
            for activation in self.polynomial_activations()
        ]
        penalties = [penalty for penalty in penalties if penalty is not None]
        if not penalties:
            return next(self.parameters()).new_zeros(())
        return torch.stack(penalties).mean()

    def polynomial_range_summary(self):
        stats = [
            activation.range_stats()
            for activation in self.polynomial_activations()
        ]
        absmax = [item["absmax"] for item in stats
                  if item["absmax"] is not None]
        outside = [item["outside_fraction"] for item in stats
                   if item["outside_fraction"] is not None]
        zero = next(self.parameters()).new_zeros(())
        return {
            "input_absmax": torch.stack(absmax).amax() if absmax else zero,
            "outside_fraction": torch.stack(outside).mean() if outside else zero,
        }

    def begin_batchnorm_recalibration(self, reset=True):
        batchnorm_state = [
            (module, module.training, module.momentum)
            for module in self.modules()
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

    def end_batchnorm_recalibration(self, state):
        self.train(state["model_training"])
        for module, was_training, momentum in state["batchnorm"]:
            module.momentum = momentum
            module.train(was_training)


def _iresnet(blocks, pretrained, **kwargs):
    model = IResNet(IBasicBlock, blocks, **kwargs)
    if pretrained:
        raise ValueError("No bundled pretrained precise-ReLU checkpoint")
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

"""IResNet curriculum from PreciseReLUAlpha10 to low-degree ReLU.

Every original channel-wise PReLU is written as

    PReLU_a(x) = a*x + (1-a)*ReLU(x)

and only its ReLU term is replaced.  Training starts with the accurate
Alpha-10 composite approximation on ``[-S, S]`` and then smoothly transitions
through independently fitted Chebyshev/minimax students.  The students are
not coefficient truncations of Alpha-10: truncating a jointly fitted power
series is not a valid lower-degree approximation.

At a completed stage the encrypted activation is polynomial-only.  The PReLU
slope is a learned plaintext channel-wise coefficient.  The default final
degree 4 needs multiplicative depth 2 for the nonlinear power schedule.
"""

import math

import torch
from torch import nn

from .iresnet import IBasicBlock, IResNet as _IResNet
from .polynomial_relu import ChebyReLU, PreciseReLUAlpha10

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


class ProgressivePrecisePReLU(nn.Module):
    """PReLU using an Alpha-10-to-low-degree ReLU curriculum."""

    is_progressive_precise_relu = True

    def __init__(self, channels, input_scale=8.0, lower_degrees=(16, 8, 4),
                 progress=0.0, initial_slope=0.25):
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        if input_scale <= 0:
            raise ValueError("input_scale must be positive")
        lower_degrees = tuple(int(degree) for degree in lower_degrees)
        if not lower_degrees:
            raise ValueError("lower_degrees must not be empty")
        if any(degree not in _SUPPORTED_LOWER_DEGREES
               for degree in lower_degrees):
            raise ValueError("lower_degrees may contain only 16, 8, and 4")
        if any(right >= left for left, right in zip(
                lower_degrees, lower_degrees[1:])):
            raise ValueError("lower_degrees must be strictly decreasing")

        self.prelu = nn.PReLU(channels, init=float(initial_slope))
        self.alpha10 = PreciseReLUAlpha10(input_scale=input_scale)
        self.students = nn.ModuleList([
            ChebyReLU(input_scale=input_scale, degree=degree)
            for degree in lower_degrees
        ])
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
            excess = torch.relu(compute_x.abs() - scale)
            self._last_range_penalty = (
                excess.square().mean()
                + 0.1 * excess.flatten(1).amax(dim=1).square().mean()
            )
            self._last_input_absmax = compute_x.detach().abs().amax()
            self._last_outside_fraction = (
                (excess.detach() > 0).float().mean())
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
                 precise_relu_lower_degrees=(16, 8, 4),
                 precise_relu_progress=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.precise_relu_input_scale = float(precise_relu_input_scale)
        self.precise_relu_lower_degrees = tuple(
            int(degree) for degree in precise_relu_lower_degrees)
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
                    lower_degrees=self.precise_relu_lower_degrees,
                    progress=float(self.polynomial_progress.item()),
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
        return len(self.precise_relu_lower_degrees)

    def set_polynomial_progress(self, progress):
        progress = min(max(float(progress), 0.0),
                       float(self.polynomial_transition_count()))
        self.polynomial_progress.fill_(progress)
        for activation in self.polynomial_activations():
            activation.set_degree_progress(progress)

    def polynomial_stage_names(self):
        return ("alpha10",) + tuple(
            "degree{}".format(degree)
            for degree in self.precise_relu_lower_degrees)

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

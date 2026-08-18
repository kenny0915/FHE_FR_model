"""Normalization-free, degree-2 IResNet for FHE face recognition.

Each residual block is linear at initialization and introduces exactly one
ciphertext-ciphertext product along its encrypted path::

    z = W1(s * x)
    a = z * (1 + beta * z)
    y = rho * shortcut(x) + alpha * W2(a)

Scaled weight standardization (SWS) and bounded scalar parameterizations are
training-time operations on plaintext parameters.  ``switch_to_deploy``
materializes them into ordinary convolutions and public constants.
"""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from .poolformer_nf import (
    BoundedScalar,
    ScaledWSConv2d,
    SymmetricBoundedScalar,
    _fold_conv_batchnorm,
    _fold_linear_batchnorm,
    _init_ws_conv,
    _replace_ws_convs,
)

__all__ = [
    "NormFreeResidualBlock",
    "NormFreeIResNet",
    "iresnet_nf8",
    "iresnet_nf12",
]


class NormFreeResidualBlock(nn.Module):
    """Two-convolution ResNet block with a bounded near-linear quadratic."""

    def __init__(self, in_channels, out_channels, stride=1, ws_eps=1e-4,
                 alpha_init=0.05, alpha_max=0.2, input_gain_init=1.0,
                 input_gain_min=0.25, input_gain_max=4.0,
                 quadratic_scale_max=0.25, range_limit=6.0,
                 range_sample_size=16384):
        super().__init__()
        if stride not in (1, 2):
            raise ValueError("NormFreeResidualBlock stride must be 1 or 2")
        if range_limit <= 0.0:
            raise ValueError("range_limit must be positive")
        if range_sample_size <= 0:
            raise ValueError("range_sample_size must be positive")

        self.range_limit = float(range_limit)
        self.range_sample_size = int(range_sample_size)
        self.input_gain = BoundedScalar(
            input_gain_init, input_gain_min, input_gain_max)
        self.quadratic_scale = SymmetricBoundedScalar(
            initial=0.0, maximum=quadratic_scale_max)
        self.alpha = BoundedScalar(alpha_init, 1e-5, alpha_max)
        self.conv1 = ScaledWSConv2d(
            in_channels, out_channels, 3, stride=1, padding=1,
            ws_eps=ws_eps)
        self.conv2 = ScaledWSConv2d(
            out_channels, out_channels, 3, stride=stride, padding=1,
            ws_eps=ws_eps)
        self.projection = (
            ScaledWSConv2d(
                in_channels, out_channels, 1, stride=stride, ws_eps=ws_eps)
            if stride != 1 or in_channels != out_channels else nn.Identity()
        )
        self.apply(_init_ws_conv)

        self.range_tracking = False
        self._last_range_tensors = None
        self.deploy = False
        self.register_buffer("deploy_beta", torch.tensor(0.0), persistent=True)
        self.register_buffer("deploy_rho", torch.tensor(1.0), persistent=True)

    def residual_coefficients(self):
        alpha = self.alpha()
        rho = torch.sqrt(torch.clamp(1.0 - alpha.square(), min=0.0))
        return rho, alpha

    def set_range_tracking(self, enabled=True):
        self.range_tracking = bool(enabled)
        if not enabled:
            self._last_range_tensors = None

    def clear_cached_tensors(self):
        self._last_range_tensors = None

    def _training_forward(self, x):
        scaled = self.input_gain().to(dtype=x.dtype) * x
        preactivation = self.conv1(scaled)
        beta = self.quadratic_scale().to(dtype=x.dtype)
        modulator = 1.0 + beta * preactivation
        product = preactivation * modulator
        branch = self.conv2(product)
        shortcut = self.projection(x)
        rho, alpha = self.residual_coefficients()
        output = (
            rho.to(dtype=x.dtype) * shortcut
            + alpha.to(dtype=x.dtype) * branch
        )
        if self.range_tracking:
            self._last_range_tensors = {
                "input": x,
                "preactivation": preactivation,
                "modulator": modulator,
                "product": product,
                "branch": branch,
                "shortcut": shortcut,
                "output": output,
            }
        return output

    def _deploy_forward(self, x):
        preactivation = self.conv1(x)
        beta = self.deploy_beta.to(dtype=x.dtype)
        activation = preactivation * (1.0 + beta * preactivation)
        shortcut = self.projection(x)
        return (
            self.deploy_rho.to(dtype=x.dtype) * shortcut
            + self.conv2(activation)
        )

    def forward(self, x):
        return self._deploy_forward(x) if self.deploy else self._training_forward(x)

    def _sample(self, value):
        flat = value.reshape(-1)
        if flat.numel() <= self.range_sample_size:
            return flat
        stride = max(flat.numel() // self.range_sample_size, 1)
        return flat[::stride][0:self.range_sample_size]

    def range_penalty(self):
        if not self._last_range_tensors:
            raise RuntimeError("Range tracking must be enabled before forward")
        limit = self.range_limit
        penalties = []
        for name in ("preactivation", "modulator", "product"):
            value = self._sample(self._last_range_tensors[name]).float()
            excess = F.relu(value.abs() - limit) / limit
            penalties.append(excess.square().mean())
        output = self._sample(self._last_range_tensors["output"]).float()
        output_rms = output.square().mean().sqrt()
        penalties.append(F.relu(output_rms / limit - 1.0).square())
        return torch.stack(penalties).mean()

    def _stats(self, value):
        value = value.detach()
        sample = self._sample(value).float()
        absolute = sample.abs()
        zero = sample.new_zeros(())
        return {
            "absmax": value.float().abs().amax() if value.numel() else zero,
            "p999": torch.quantile(absolute, 0.999) if absolute.numel() else zero,
            "rms": sample.square().mean().sqrt() if sample.numel() else zero,
        }

    def range_summary(self):
        if not self._last_range_tensors:
            return {}
        summary = {}
        for name, value in self._last_range_tensors.items():
            for metric, result in self._stats(value).items():
                summary[f"{name}_{metric}"] = result
        rho, alpha = self.residual_coefficients()
        summary.update({
            "alpha": alpha.detach(),
            "rho": rho.detach(),
            "input_gain": self.input_gain().detach(),
            "quadratic_scale": self.quadratic_scale().detach(),
        })
        return summary

    def switch_to_deploy(self):
        """Fold SWS and train-time scalar gains into the inference block."""
        if self.deploy:
            return self
        with torch.no_grad():
            input_gain = float(self.input_gain().item())
            beta = float(self.quadratic_scale().item())
            rho, alpha = self.residual_coefficients()
            self.deploy_beta.fill_(beta)
            self.conv1 = self.conv1.to_conv2d(input_scale=input_gain)
            self.conv2 = self.conv2.to_conv2d(
                output_scale=float(alpha.item()))
            if isinstance(self.projection, ScaledWSConv2d):
                self.projection = self.projection.to_conv2d(
                    output_scale=float(rho.item()))
                self.deploy_rho.fill_(1.0)
            else:
                self.deploy_rho.fill_(float(rho.item()))
        del self.input_gain
        del self.quadratic_scale
        del self.alpha
        self.deploy = True
        self._last_range_tensors = None
        return self


class NormFreeIResNet(nn.Module):
    """Four-stage normalization-free IResNet face-embedding backbone."""

    def __init__(self, layers=(2, 2, 6, 2), channels=(64, 128, 256, 512),
                 num_classes=512, face_embedding=True, fp16=False,
                 ws_eps=1e-4, alpha_init=0.05, alpha_max=0.2,
                 input_gain_init=1.0, quadratic_scale_max=0.25,
                 range_limit=6.0, range_sample_size=16384, **kwargs):
        super().__init__()
        del kwargs
        if fp16:
            raise ValueError(
                "NormFreeIResNet must first be trained in FP32; set fp16=False")
        if len(layers) != 4 or len(channels) != 4:
            raise ValueError("NormFreeIResNet expects four stages")
        self.num_classes = int(num_classes)
        self.face_embedding = bool(face_embedding)
        self.fp16 = False
        self.stem = ScaledWSConv2d(
            3, channels[0], 3, stride=1, padding=1, ws_eps=ws_eps)
        _init_ws_conv(self.stem)

        stages = []
        in_channels = channels[0]
        for depth, out_channels in zip(layers, channels):
            blocks = []
            for block_index in range(depth):
                blocks.append(NormFreeResidualBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=2 if block_index == 0 else 1,
                    ws_eps=ws_eps,
                    alpha_init=alpha_init,
                    alpha_max=alpha_max,
                    input_gain_init=input_gain_init,
                    quadratic_scale_max=quadratic_scale_max,
                    range_limit=range_limit,
                    range_sample_size=range_sample_size,
                ))
                in_channels = out_channels
            stages.append(nn.Sequential(*blocks))
        self.stages = nn.ModuleList(stages)

        if self.face_embedding:
            self.head = nn.Sequential(
                ScaledWSConv2d(
                    channels[-1], channels[-1], kernel_size=7,
                    ws_eps=ws_eps),
                nn.BatchNorm2d(channels[-1]),
                nn.Flatten(),
                nn.Linear(channels[-1], num_classes, bias=False),
                nn.BatchNorm1d(num_classes),
            )
            _init_ws_conv(self.head[0])
            nn.init.trunc_normal_(self.head[3].weight, std=0.02)
        else:
            self.head = nn.Linear(channels[-1], num_classes)
            nn.init.trunc_normal_(self.head.weight, std=0.02)
            nn.init.zeros_(self.head.bias)
        self.deployed = False

    def nf_blocks(self):
        return [module for module in self.modules()
                if isinstance(module, NormFreeResidualBlock)]

    def set_nf_range_tracking(self, enabled=True):
        for block in self.nf_blocks():
            block.set_range_tracking(enabled)

    def clear_nf_cached_tensors(self):
        for block in self.nf_blocks():
            block.clear_cached_tensors()

    def nf_range_penalty(self):
        losses = [block.range_penalty() for block in self.nf_blocks()]
        if not losses:
            return next(self.parameters()).new_zeros(())
        return torch.stack(losses).mean()

    def nf_range_summary(self):
        return {
            f"block_{index:02d}": block.range_summary()
            for index, block in enumerate(self.nf_blocks())
        }

    def forward_features(self, x):
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        if self.face_embedding:
            return self.head(x)
        return self.head(x.mean(dim=(-2, -1)))

    def switch_to_deploy(self, inplace=False):
        """Return an eval graph with ordinary convolutions and no BatchNorm."""
        model = self if inplace else copy.deepcopy(self)
        if model.training:
            raise RuntimeError("Call eval() before switch_to_deploy()")
        if model.deployed:
            return model
        for block in model.nf_blocks():
            block.switch_to_deploy()
        _replace_ws_convs(model)
        if model.face_embedding:
            model.head[0] = _fold_conv_batchnorm(
                model.head[0], model.head[1])
            model.head[1] = nn.Identity()
            model.head[3] = _fold_linear_batchnorm(
                model.head[3], model.head[4])
            model.head[4] = nn.Identity()
        model.deployed = True
        return model


def iresnet_nf12(pretrained=False, **kwargs):
    """Normalization-free IResNet with stage depths ``[2, 2, 6, 2]``."""
    if pretrained:
        raise ValueError("iresnet_nf12 is designed for training from scratch")
    return NormFreeIResNet(
        layers=(2, 2, 6, 2), channels=(64, 128, 256, 512), **kwargs)


def iresnet_nf8(pretrained=False, **kwargs):
    """Smaller normalization-free IResNet ablation ``[1, 2, 4, 1]``."""
    if pretrained:
        raise ValueError("iresnet_nf8 is designed for training from scratch")
    return NormFreeIResNet(
        layers=(1, 2, 4, 1), channels=(64, 128, 256, 512), **kwargs)

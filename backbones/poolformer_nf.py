"""Normalization-free PoolFormer for low-depth FHE face recognition.

The encrypted backbone contains only convolutions, additions, plaintext scalar
multiplications, average pooling, and one ciphertext-ciphertext product per
block.  Scaled weight standardization (SWS) is evaluated on plaintext weights
during training and can be materialized into ordinary convolutions for export.

The gate is deliberately close to linear at initialization::

    u = W_u(s x)
    v = 1 + beta W_v(s x)
    g = u * v

``beta`` starts at zero.  The residual coefficient and token-mixing strength
are bounded scalars, preventing a block from abruptly amplifying its input.
"""

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _logit(value):
    value = float(value)
    return math.log(value / (1.0 - value))


class BoundedScalar(nn.Module):
    """A scalar constrained to ``[minimum, maximum]`` by a sigmoid."""

    def __init__(self, initial, minimum, maximum):
        super().__init__()
        self.exclude_from_weight_decay = True
        if not minimum < initial < maximum:
            raise ValueError(
                f"Expected {minimum} < initial < {maximum}, got {initial}")
        self.minimum = float(minimum)
        self.maximum = float(maximum)
        ratio = (float(initial) - self.minimum) / (
            self.maximum - self.minimum)
        self.raw = nn.Parameter(torch.tensor(_logit(ratio)))

    def forward(self):
        return self.minimum + (self.maximum - self.minimum) * torch.sigmoid(
            self.raw)


class SymmetricBoundedScalar(nn.Module):
    """A scalar constrained to ``[-maximum, maximum]`` by ``tanh``."""

    def __init__(self, initial=0.0, maximum=0.25):
        super().__init__()
        self.exclude_from_weight_decay = True
        if maximum <= 0 or abs(initial) >= maximum:
            raise ValueError("initial must be strictly inside the bound")
        self.maximum = float(maximum)
        normalized = float(initial) / self.maximum
        self.raw = nn.Parameter(torch.tensor(math.atanh(normalized)))

    def forward(self):
        return self.maximum * torch.tanh(self.raw)


class ScaledWSConv2d(nn.Conv2d):
    """Conv2d with per-output-filter scaled weight standardization.

    SWS is a parameterization of plaintext weights, not an activation
    normalization.  :meth:`to_conv2d` materializes the effective weight so the
    exported graph contains a regular convolution only.
    """

    def __init__(self, *args, ws_eps=1e-4, **kwargs):
        super().__init__(*args, **kwargs)
        if ws_eps <= 0:
            raise ValueError("ws_eps must be positive")
        self.ws_eps = float(ws_eps)
        self.gain = nn.Parameter(torch.ones(self.out_channels))

    def effective_weight(self):
        weight = self.weight.float()
        centered = weight - weight.mean(dim=(1, 2, 3), keepdim=True)
        variance = centered.square().mean(dim=(1, 2, 3), keepdim=True)
        fan_in = self.weight[0].numel()
        scale = self.gain.float().view(-1, 1, 1, 1) / math.sqrt(fan_in)
        effective = centered * torch.rsqrt(variance + self.ws_eps) * scale
        return effective.to(dtype=self.weight.dtype)

    def forward(self, x):
        return F.conv2d(
            x, self.effective_weight(), self.bias, self.stride, self.padding,
            self.dilation, self.groups)

    def to_conv2d(self, input_scale=1.0, output_scale=1.0):
        """Return an ordinary convolution with scalar gains folded in."""
        converted = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
            bias=self.bias is not None,
            padding_mode=self.padding_mode,
            device=self.weight.device,
            dtype=self.weight.dtype,
        )
        total_scale = float(input_scale) * float(output_scale)
        with torch.no_grad():
            converted.weight.copy_(self.effective_weight() * total_scale)
            if self.bias is not None:
                converted.bias.copy_(self.bias * float(output_scale))
        converted.train(self.training)
        return converted


def _init_ws_conv(module, std=0.02):
    if isinstance(module, ScaledWSConv2d):
        nn.init.trunc_normal_(module.weight, std=std)
        nn.init.ones_(module.gain)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class StableTokenMixer(nn.Module):
    """Convex local averaging, hence non-expansive in the infinity norm."""

    def __init__(self, pool_size=3, tau_init=0.1):
        super().__init__()
        self.pool = nn.AvgPool2d(
            pool_size, stride=1, padding=pool_size // 2,
            count_include_pad=False)
        self.tau = BoundedScalar(tau_init, 1e-4, 1.0 - 1e-4)

    def forward(self, x):
        tau = self.tau().to(dtype=x.dtype)
        return (1.0 - tau) * x + tau * self.pool(x)


class NormFreeGatedBlock(nn.Module):
    """One stable token mixer and one near-linear quadratic residual branch."""

    def __init__(self, dim, pool_size=3, ws_eps=1e-4, tau_init=0.1,
                 alpha_init=0.05, alpha_max=0.2, input_gain_init=1.0,
                 input_gain_min=0.25, input_gain_max=4.0,
                 modulator_scale_max=0.25, range_limit=6.0,
                 range_sample_size=16384,
                 initial_modulation_progress=1.0):
        super().__init__()
        if range_limit <= 0:
            raise ValueError("range_limit must be positive")
        if range_sample_size <= 0:
            raise ValueError("range_sample_size must be positive")
        if not 0.0 <= initial_modulation_progress <= 1.0:
            raise ValueError(
                "initial_modulation_progress must be in [0, 1]")
        self.dim = int(dim)
        self.range_limit = float(range_limit)
        self.range_sample_size = int(range_sample_size)
        self.token_mixer = StableTokenMixer(pool_size, tau_init)
        self.input_gain = BoundedScalar(
            input_gain_init, input_gain_min, input_gain_max)
        self.modulator_scale = SymmetricBoundedScalar(
            initial=0.0, maximum=modulator_scale_max)
        self.alpha = BoundedScalar(alpha_init, 1e-5, alpha_max)
        self.conv_u = ScaledWSConv2d(
            dim, dim, 1, bias=False, ws_eps=ws_eps)
        self.conv_v = ScaledWSConv2d(
            dim, dim, 1, bias=False, ws_eps=ws_eps)
        self.conv_out = ScaledWSConv2d(
            dim, dim, 1, bias=False, ws_eps=ws_eps)
        self.apply(_init_ws_conv)

        self.range_tracking = False
        self._last_range_tensors = None
        self.deploy = False
        self.register_buffer("deploy_tau", torch.tensor(0.0), persistent=True)
        self.register_buffer("deploy_rho", torch.tensor(1.0), persistent=True)
        self.register_buffer(
            "modulation_progress",
            torch.tensor(float(initial_modulation_progress)),
            persistent=True,
        )

    def residual_coefficients(self):
        alpha = self.alpha()
        # Convex residual mixing prevents twelve positively correlated
        # branches from accumulating coefficients greater than one.
        rho = 1.0 - alpha
        return rho, alpha

    def set_modulation_progress(self, progress):
        progress = float(progress)
        if not 0.0 <= progress <= 1.0:
            raise ValueError("modulation progress must be in [0, 1]")
        self.modulation_progress.fill_(progress)

    def modulation_coefficient(self):
        return self.modulator_scale() * self.modulation_progress

    def set_range_tracking(self, enabled=True):
        self.range_tracking = bool(enabled)
        if not enabled:
            self._last_range_tensors = None

    def clear_cached_tensors(self):
        self._last_range_tensors = None

    def _training_forward(self, x):
        mixed = self.token_mixer(x)
        scaled = self.input_gain().to(dtype=x.dtype) * mixed
        u = self.conv_u(scaled)
        modulation = self.modulation_coefficient().to(dtype=x.dtype)
        v = 1.0 + modulation * self.conv_v(scaled)
        product = u * v
        branch = self.conv_out(product)
        rho, alpha = self.residual_coefficients()
        output = rho.to(dtype=x.dtype) * x + alpha.to(dtype=x.dtype) * branch
        if self.range_tracking:
            self._last_range_tensors = {
                "input": x,
                "mixed": mixed,
                "operand_u": u,
                "operand_v": v,
                "product": product,
                "branch": branch,
                "output": output,
            }
        return output

    def _deploy_forward(self, x):
        tau = self.deploy_tau.to(dtype=x.dtype)
        mixed = (1.0 - tau) * x + tau * self.token_mixer.pool(x)
        u = self.conv_u(mixed)
        v = 1.0 + self.conv_v(mixed)
        branch = self.conv_out(u * v)
        return self.deploy_rho.to(dtype=x.dtype) * x + branch

    def forward(self, x):
        return self._deploy_forward(x) if self.deploy else self._training_forward(x)

    def range_penalty(self):
        if not self._last_range_tensors:
            raise RuntimeError("Range tracking must be enabled before forward")
        limit = self.range_limit
        penalties = []
        for name in ("operand_u", "operand_v", "product"):
            full_value = self._last_range_tensors[name].float()
            value = self._sample(full_value)
            # Log compression is training-only. It preserves a zero penalty
            # inside [-limit, limit] but cannot create an enormous auxiliary
            # gradient when a rare quadratic outlier is still finite.
            excess = F.relu(
                torch.log1p(value.abs() / limit) - math.log(2.0))
            max_excess = F.relu(
                torch.log1p(full_value.abs().amax() / limit)
                - math.log(2.0))
            penalties.append(
                excess.square().mean() + 0.1 * max_excess.square())
        output = self._sample(self._last_range_tensors["output"]).float()
        output_scale = output.detach().abs().amax().clamp_min(1e-12)
        output_rms = output_scale * (
            (output / output_scale).square().mean().sqrt())
        # A soft scale guard complements the tail penalty without trying to
        # force every stage to have identical variance.
        rms_excess = F.relu(
            torch.log1p(output_rms / limit) - math.log(2.0))
        penalties.append(rms_excess.square())
        return torch.stack(penalties).mean()

    def _sample(self, value):
        flat = value.reshape(-1)
        if flat.numel() <= self.range_sample_size:
            return flat
        stride = max(flat.numel() // self.range_sample_size, 1)
        return flat[::stride][0:self.range_sample_size]

    def _stats(self, value):
        value = value.detach()
        sample = self._sample(value).float()
        flat = sample.abs()
        p999 = torch.quantile(flat, 0.999) if flat.numel() else sample.new_zeros(())
        return {
            "absmax": value.float().abs().max() if value.numel()
            else sample.new_zeros(()),
            "p999": p999,
            "rms": sample.square().mean().sqrt(),
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
            "tau": self.token_mixer.tau().detach(),
            "alpha": alpha.detach(),
            "rho": rho.detach(),
            "input_gain": self.input_gain().detach(),
            "modulator_scale": self.modulator_scale().detach(),
            "modulation_progress": self.modulation_progress.detach(),
            "effective_modulation": self.modulation_coefficient().detach(),
        })
        return summary

    def switch_to_deploy(self):
        """Materialize SWS and fold all train-time scalar branch gains."""
        if self.deploy:
            return self
        with torch.no_grad():
            tau = float(self.token_mixer.tau().item())
            input_gain = float(self.input_gain().item())
            modulation = float(self.modulation_coefficient().item())
            rho, alpha = self.residual_coefficients()
            self.deploy_tau.fill_(tau)
            self.deploy_rho.fill_(float(rho.item()))
            self.conv_u = self.conv_u.to_conv2d(input_scale=input_gain)
            self.conv_v = self.conv_v.to_conv2d(
                input_scale=input_gain,
                output_scale=modulation)
            self.conv_out = self.conv_out.to_conv2d(
                output_scale=float(alpha.item()))
        del self.token_mixer.tau
        del self.input_gain
        del self.modulator_scale
        del self.alpha
        self.deploy = True
        self._last_range_tensors = None
        return self


class PatchEmbed(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2,
                 padding=1, ws_eps=1e-4):
        super().__init__()
        self.proj = ScaledWSConv2d(
            in_channels, out_channels, kernel_size, stride=stride,
            padding=padding, bias=False, ws_eps=ws_eps)
        _init_ws_conv(self.proj)

    def forward(self, x):
        return self.proj(x)


def _replace_ws_convs(module):
    for name, child in list(module.named_children()):
        if isinstance(child, ScaledWSConv2d):
            setattr(module, name, child.to_conv2d())
        else:
            _replace_ws_convs(child)


def _fold_conv_batchnorm(conv, batchnorm):
    if conv.training or batchnorm.training:
        raise RuntimeError("Convolution/BatchNorm folding requires eval mode")
    fused = nn.Conv2d(
        conv.in_channels, conv.out_channels, conv.kernel_size,
        stride=conv.stride, padding=conv.padding, dilation=conv.dilation,
        groups=conv.groups, bias=True, padding_mode=conv.padding_mode,
        device=conv.weight.device, dtype=conv.weight.dtype)
    weight = conv.weight.detach()
    bias = (conv.bias.detach() if conv.bias is not None
            else weight.new_zeros(conv.out_channels))
    bn_weight = (batchnorm.weight.detach() if batchnorm.affine
                 else weight.new_ones(conv.out_channels))
    bn_bias = (batchnorm.bias.detach() if batchnorm.affine
               else weight.new_zeros(conv.out_channels))
    scale = bn_weight * torch.rsqrt(
        batchnorm.running_var.detach() + batchnorm.eps)
    with torch.no_grad():
        fused.weight.copy_(weight * scale.view(-1, 1, 1, 1))
        fused.bias.copy_((bias - batchnorm.running_mean.detach()) * scale
                         + bn_bias)
    fused.eval()
    return fused


def _fold_linear_batchnorm(linear, batchnorm):
    if linear.training or batchnorm.training:
        raise RuntimeError("Linear/BatchNorm folding requires eval mode")
    fused = nn.Linear(
        linear.in_features, linear.out_features, bias=True,
        device=linear.weight.device, dtype=linear.weight.dtype)
    weight = linear.weight.detach()
    bias = (linear.bias.detach() if linear.bias is not None
            else weight.new_zeros(linear.out_features))
    bn_weight = (batchnorm.weight.detach() if batchnorm.affine
                 else weight.new_ones(linear.out_features))
    bn_bias = (batchnorm.bias.detach() if batchnorm.affine
               else weight.new_zeros(linear.out_features))
    scale = bn_weight * torch.rsqrt(
        batchnorm.running_var.detach() + batchnorm.eps)
    with torch.no_grad():
        fused.weight.copy_(weight * scale.view(-1, 1))
        fused.bias.copy_((bias - batchnorm.running_mean.detach()) * scale
                         + bn_bias)
    fused.eval()
    return fused


class NormFreePoolFormer(nn.Module):
    """Four-stage normalization-free face-recognition backbone."""

    def __init__(self, layers=(2, 2, 6, 2), embed_dims=(64, 128, 256, 512),
                 num_classes=512, face_embedding=True, fp16=False,
                 pool_size=3, ws_eps=1e-4, tau_init=0.1,
                 alpha_init=0.05, alpha_max=0.2, input_gain_init=1.0,
                 input_gain_min=0.25, input_gain_max=4.0,
                 modulator_scale_max=0.25, range_limit=6.0,
                 range_sample_size=16384,
                 initial_modulation_progress=1.0,
                 learnable_ws_gain=False, **kwargs):
        super().__init__()
        if fp16:
            raise ValueError(
                "NormFreePoolFormer must first be trained in FP32; set fp16=False")
        if len(layers) != 4 or len(embed_dims) != 4:
            raise ValueError("NF PoolFormer expects four stages")
        self.num_classes = int(num_classes)
        self.face_embedding = bool(face_embedding)
        self.fp16 = False
        self.patch_embed = PatchEmbed(3, embed_dims[0], ws_eps=ws_eps)

        network = []
        for stage, (depth, dim) in enumerate(zip(layers, embed_dims)):
            blocks = [NormFreeGatedBlock(
                dim=dim,
                pool_size=pool_size,
                ws_eps=ws_eps,
                tau_init=tau_init,
                alpha_init=alpha_init,
                alpha_max=alpha_max,
                input_gain_init=input_gain_init,
                input_gain_min=input_gain_min,
                input_gain_max=input_gain_max,
                modulator_scale_max=modulator_scale_max,
                range_limit=range_limit,
                range_sample_size=range_sample_size,
                initial_modulation_progress=initial_modulation_progress,
            ) for _ in range(depth)]
            network.append(nn.Sequential(*blocks))
            if stage < 3:
                network.append(PatchEmbed(
                    dim, embed_dims[stage + 1], ws_eps=ws_eps))
        self.network = nn.ModuleList(network)

        if self.face_embedding:
            self.head = nn.Sequential(
                ScaledWSConv2d(
                    embed_dims[-1], embed_dims[-1], kernel_size=7,
                    bias=False, ws_eps=ws_eps),
                nn.BatchNorm2d(embed_dims[-1]),
                nn.Flatten(),
                nn.Linear(embed_dims[-1], num_classes, bias=False),
                nn.BatchNorm1d(num_classes),
            )
            _init_ws_conv(self.head[0])
            nn.init.trunc_normal_(self.head[3].weight, std=0.02)
        else:
            self.head = nn.Linear(embed_dims[-1], num_classes)
            nn.init.trunc_normal_(self.head.weight, std=0.02)
            nn.init.zeros_(self.head.bias)
        if not learnable_ws_gain:
            for module in self.modules():
                if isinstance(module, ScaledWSConv2d):
                    module.gain.requires_grad_(False)
        self.deployed = False

    def nf_blocks(self):
        return [module for module in self.modules()
                if isinstance(module, NormFreeGatedBlock)]

    def set_nf_range_tracking(self, enabled=True):
        for block in self.nf_blocks():
            block.set_range_tracking(enabled)

    def set_nf_modulation_progresses(self, progresses, order="forward"):
        blocks = self.nf_blocks()
        progresses = tuple(float(value) for value in progresses)
        if len(progresses) != len(blocks):
            raise ValueError(
                f"Expected {len(blocks)} modulation progresses, got "
                f"{len(progresses)}")
        if order == "reverse":
            blocks = list(reversed(blocks))
        elif order != "forward":
            raise ValueError("NF modulation order must be 'forward' or 'reverse'")
        for block, progress in zip(blocks, progresses):
            block.set_modulation_progress(progress)

    def nf_modulation_group_count(self):
        return len(self.nf_blocks())

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
        x = self.patch_embed(x)
        for module in self.network:
            x = module(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        if self.face_embedding:
            return self.head(x)
        return self.head(x.mean(dim=(-2, -1)))

    def switch_to_deploy(self, inplace=False):
        """Return an eval model containing no SWS or bounded parameters."""
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


def poolformer_nf12(pretrained=False, **kwargs):
    """FHEPoolFormer-NF12: stage depths ``[2, 2, 6, 2]``."""
    if pretrained:
        raise ValueError("poolformer_nf12 is designed for training from scratch")
    model = NormFreePoolFormer(
        layers=(2, 2, 6, 2),
        embed_dims=(64, 128, 256, 512),
        **kwargs,
    )
    model.default_cfg = {}
    return model


def poolformer_nf8(pretrained=False, **kwargs):
    """Smaller ablation model with stage depths ``[1, 2, 4, 1]``."""
    if pretrained:
        raise ValueError("poolformer_nf8 is designed for training from scratch")
    model = NormFreePoolFormer(
        layers=(1, 2, 4, 1),
        embed_dims=(64, 128, 256, 512),
        **kwargs,
    )
    model.default_cfg = {}
    return model

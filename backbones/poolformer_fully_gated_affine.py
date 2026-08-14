"""Calibrated LayerNorm-to-affine conversion for fully gated PoolFormer.

The accepted LayerNorm model is preserved exactly at the start of fine-tuning.
Representative inputs are then used to fit, independently for every channel,
the least-squares affine approximation ``a * x + b`` to each LayerNorm output.
During fine-tuning the LayerNorm teacher decays linearly and the calibrated
affine student takes over.  The final inference graph has no data-dependent
normalization, division, reciprocal square root, or additional ciphertext-
ciphertext multiplication.
"""

from collections import OrderedDict

import torch
import torch.distributed as dist
import torch.nn as nn

from .poolformer_fully_gated import FullyGatedPoolFormer, LayerNorm2d


class ChannelAffine2d(nn.Module):
    """Learned fixed per-channel affine transform for ``[B, C, H, W]``."""

    exclude_from_weight_decay = True

    def __init__(self, num_channels):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))

    def forward(self, x):
        return (
            self.weight.view(1, -1, 1, 1) * x
            + self.bias.view(1, -1, 1, 1)
        )


class ProgressiveAffineNorm2d(nn.Module):
    """Progressively replace LayerNorm with a calibrated channel affine map."""

    def __init__(self, num_channels, ln_eps=1e-6):
        super().__init__()
        self.ln = LayerNorm2d(num_channels, eps=ln_eps)
        self.affine = ChannelAffine2d(num_channels)
        self.register_buffer("gamma", torch.ones(1))
        self._gamma = 1.0
        self._collect_calibration = False

        # Float64 accumulation keeps the centered least-squares calculation
        # stable when a site has a large number of spatial samples.
        self.register_buffer(
            "calibration_count", torch.zeros((), dtype=torch.float64),
            persistent=False)
        for name in ("sum_x", "sum_y", "sum_xx", "sum_xy", "sum_yy"):
            self.register_buffer(
                f"calibration_{name}",
                torch.zeros(num_channels, dtype=torch.float64),
                persistent=False,
            )

    def set_progress(self, current_step, total_steps):
        if total_steps <= 0:
            gamma = 0.0
        else:
            gamma = 1.0 - float(current_step) / float(total_steps)
            gamma = min(1.0, max(0.0, gamma))
        self._gamma = gamma
        self.gamma.fill_(gamma)

    def _load_from_state_dict(self, *args, **kwargs):
        super()._load_from_state_dict(*args, **kwargs)
        self._gamma = float(self.gamma.detach().item())

    def begin_calibration(self):
        self.calibration_count.zero_()
        self.calibration_sum_x.zero_()
        self.calibration_sum_y.zero_()
        self.calibration_sum_xx.zero_()
        self.calibration_sum_xy.zero_()
        self.calibration_sum_yy.zero_()
        self._collect_calibration = True

    @torch.no_grad()
    def _accumulate_calibration(self, x, target):
        x64 = x.detach().to(dtype=torch.float64)
        target64 = target.detach().to(dtype=torch.float64)
        reduce_dims = (0, 2, 3)
        self.calibration_count.add_(x.shape[0] * x.shape[2] * x.shape[3])
        self.calibration_sum_x.add_(x64.sum(dim=reduce_dims))
        self.calibration_sum_y.add_(target64.sum(dim=reduce_dims))
        self.calibration_sum_xx.add_((x64 * x64).sum(dim=reduce_dims))
        self.calibration_sum_xy.add_((x64 * target64).sum(dim=reduce_dims))
        self.calibration_sum_yy.add_((target64 * target64).sum(dim=reduce_dims))

    @torch.no_grad()
    def finish_calibration(self, ridge=1e-6, distributed=True):
        """Fit ``affine(x)`` to LayerNorm outputs and return fit diagnostics."""
        self._collect_calibration = False
        statistics = (
            self.calibration_count,
            self.calibration_sum_x,
            self.calibration_sum_y,
            self.calibration_sum_xx,
            self.calibration_sum_xy,
            self.calibration_sum_yy,
        )
        if (distributed and dist.is_available() and dist.is_initialized()
                and dist.get_world_size() > 1):
            for statistic in statistics:
                dist.all_reduce(statistic, op=dist.ReduceOp.SUM)

        count = self.calibration_count
        if count.item() <= 0:
            raise RuntimeError("Affine normalization calibration saw no samples")
        mean_x = self.calibration_sum_x / count
        mean_y = self.calibration_sum_y / count
        centered_xx = (
            self.calibration_sum_xx
            - self.calibration_sum_x.square() / count
        ).clamp_min(0.0)
        centered_xy = (
            self.calibration_sum_xy
            - self.calibration_sum_x * self.calibration_sum_y / count
        )
        denominator = centered_xx + float(ridge) * count
        scale = centered_xy / denominator.clamp_min(torch.finfo(torch.float64).tiny)
        bias = mean_y - scale * mean_x
        self.affine.weight.copy_(scale.to(self.affine.weight))
        self.affine.bias.copy_(bias.to(self.affine.bias))

        residual_ss = (
            self.calibration_sum_yy
            + scale.square() * self.calibration_sum_xx
            + count * bias.square()
            - 2.0 * scale * self.calibration_sum_xy
            - 2.0 * bias * self.calibration_sum_y
            + 2.0 * scale * bias * self.calibration_sum_x
        ).clamp_min(0.0)
        centered_target_ss = (
            self.calibration_sum_yy
            - self.calibration_sum_y.square() / count
        ).clamp_min(torch.finfo(torch.float64).tiny)
        relative_rmse = torch.sqrt(residual_ss / centered_target_ss)
        return {
            "count": int(count.item()),
            "scale_absmax": float(scale.abs().max().item()),
            "bias_absmax": float(bias.abs().max().item()),
            "relative_rmse_mean": float(relative_rmse.mean().item()),
            "relative_rmse_max": float(relative_rmse.max().item()),
        }

    def forward(self, x):
        if self._gamma <= 0.0 and not self.training:
            return self.affine(x)

        teacher = self.ln(x)
        if self._collect_calibration:
            self._accumulate_calibration(x, teacher)
        if self._gamma >= 1.0:
            return teacher

        student = self.affine(x)
        # Keep the teacher parameters in DDP's training graph at gamma zero.
        gamma = self.gamma.to(dtype=x.dtype)
        return gamma * teacher + (1.0 - gamma) * student


class AffineFullyGatedPoolFormer(FullyGatedPoolFormer):
    """Fully gated PoolFormer with all LayerNorm sites under affine conversion."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._replace_layer_norms(self)

    @classmethod
    def _replace_layer_norms(cls, parent):
        for name, child in tuple(parent.named_children()):
            if isinstance(child, LayerNorm2d):
                replacement = ProgressiveAffineNorm2d(
                    child.weight.numel(), ln_eps=child.eps)
                with torch.no_grad():
                    replacement.ln.weight.copy_(child.weight)
                    replacement.ln.bias.copy_(child.bias)
                setattr(parent, name, replacement)
            else:
                cls._replace_layer_norms(child)

    def affine_norm_modules(self):
        return tuple(
            module for module in self.modules()
            if isinstance(module, ProgressiveAffineNorm2d)
        )

    def begin_affine_calibration(self):
        for module in self.affine_norm_modules():
            module.begin_calibration()

    def finish_affine_calibration(self, ridge=1e-6, distributed=True):
        return tuple(
            module.finish_calibration(ridge=ridge, distributed=distributed)
            for module in self.affine_norm_modules()
        )

    def fold_affine_norms_for_inference(self):
        """Remove the LayerNorm teachers from a fully converted eval model."""
        if self.training:
            raise RuntimeError("Call eval() before folding affine normalization")
        incomplete = [
            module for module in self.affine_norm_modules()
            if module._gamma > 0.0
        ]
        if incomplete:
            raise RuntimeError(
                "All affine normalization sites must be fully converted "
                "before folding")

        def replace(parent):
            for name, child in tuple(parent.named_children()):
                if isinstance(child, ProgressiveAffineNorm2d):
                    setattr(parent, name, child.affine)
                else:
                    replace(child)

        replace(self)
        return self

    def load_backbone_init_state_dict(self, state_dict):
        """Strictly warm-start the LayerNorm teacher from the accepted model."""
        source = OrderedDict(state_dict)
        if source and all(key.startswith("module.") for key in source):
            source = OrderedDict(
                (key[len("module."):], value)
                for key, value in source.items())

        target = self.state_dict()
        affine_norm_names = {
            name for name, module in self.named_modules()
            if isinstance(module, ProgressiveAffineNorm2d)
        }
        source_to_target = {}
        for target_key in target:
            matched_name = next(
                (name for name in affine_norm_names
                 if target_key.startswith(name + ".")),
                None,
            )
            if matched_name is None:
                source_to_target[target_key] = target_key
                continue
            suffix = target_key[len(matched_name) + 1:]
            if suffix in ("ln.weight", "ln.bias"):
                source_to_target[
                    f"{matched_name}.{suffix[len('ln.'):]}"
                ] = target_key

        missing = sorted(set(source_to_target).difference(source))
        unexpected = sorted(set(source).difference(source_to_target))
        if missing or unexpected:
            raise RuntimeError(
                "LayerNorm backbone initialization is not an exact architecture "
                f"match; missing={missing}, unexpected={unexpected}")

        translated = OrderedDict(
            (target_key, source[source_key])
            for source_key, target_key in source_to_target.items()
        )
        for target_key, value in translated.items():
            if value.shape != target[target_key].shape:
                raise RuntimeError(
                    f"Shape mismatch for {target_key}: checkpoint "
                    f"{tuple(value.shape)} != model {tuple(target[target_key].shape)}")

        incompatible = super().load_state_dict(translated, strict=False)
        if incompatible.unexpected_keys:
            raise RuntimeError(
                "Unexpected translated initialization keys: "
                f"{incompatible.unexpected_keys}")
        return incompatible


def poolformer_fully_gated_affine_s24(pretrained=False, **kwargs):
    """Warm-startable affine-normalized PoolFormer-S24."""
    if pretrained:
        raise ValueError(
            "poolformer_fully_gated_affine_s24 uses backbone_init instead of "
            "the pretrained factory flag")
    model = AffineFullyGatedPoolFormer(
        layers=[4, 4, 12, 4],
        embed_dims=[64, 128, 320, 512],
        ffn_expands=[2.0, 2.0, 2.0, 2.0],
        downsamples=[True, True, True, True],
        layer_scale_init_value=0.0,
        **kwargs,
    )
    model.default_cfg = {}
    return model

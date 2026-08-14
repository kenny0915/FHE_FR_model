"""PRepBN conversion of the FP32 fully gated PoolFormer-S24.

The LayerNorm teacher is exactly the channel-wise, per-spatial-position
normalization used by :mod:`poolformer_fully_gated`.  During training its
contribution decays linearly while RepBN takes over.  At the end of the
transition every normalization is a fixed channel-wise affine map at
inference, so it contains no encrypted-data-dependent division or square root.
"""

from collections import OrderedDict

import torch
import torch.nn as nn

from .poolformer_fully_gated import FullyGatedPoolFormer, LayerNorm2d


class ProgressiveRepBatchNorm2d(nn.Module):
    """Progressively replace an exact ``LayerNorm2d`` teacher with RepBN.

    The RepBN branch follows Guo et al. (ICML 2024):

        RepBN(x) = BN(x) + eta * x

    ``gamma`` is one at initialization and decays to zero.  The Python mirror
    avoids reading a CUDA scalar in every normalization layer and forward pass.
    """

    def __init__(self, num_channels, ln_eps=1e-6, bn_eps=1e-5,
                 bn_momentum=0.1, eta_init=0.0):
        super().__init__()
        self.ln = LayerNorm2d(num_channels, eps=ln_eps)
        self.bn = nn.BatchNorm2d(
            num_channels, eps=bn_eps, momentum=bn_momentum)
        # The paper and its official implementation use one learned residual
        # coefficient per normalization layer.  It remains a plaintext
        # constant after training and can be merged into the affine scale.
        self.eta = nn.Parameter(torch.tensor(float(eta_init)))
        self.register_buffer("gamma", torch.ones(1))
        self._gamma = 1.0

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
        # Loading happens only at initialization/resume, so this one scalar
        # device synchronization does not affect training throughput.
        self._gamma = float(self.gamma.detach().item())

    def equivalent_affine(self):
        """Return the frozen-inference ``scale, bias`` for RepBN.

        For an input ``x``, pure RepBN is exactly
        ``scale[None,:,None,None] * x + bias[None,:,None,None]``.
        """
        inverse_std = torch.rsqrt(self.bn.running_var + self.bn.eps)
        scale = self.bn.weight * inverse_std + self.eta
        bias = self.bn.bias - self.bn.weight * self.bn.running_mean * inverse_std
        return scale, bias

    def forward(self, x):
        # Preserve the accepted checkpoint exactly at the start and omit the
        # LayerNorm computation entirely in the final deployment graph.
        if self._gamma >= 1.0:
            return self.ln(x)

        rep_bn = self.bn(x) + self.eta * x
        if self._gamma <= 0.0 and not self.training:
            return rep_bn

        # Keep the teacher parameters in the DDP graph after gamma reaches
        # zero.  Their gradients are exactly zero, while eval/deployment omits
        # the teacher branch above.
        gamma = self.gamma.to(dtype=x.dtype)
        return gamma * self.ln(x) + (1.0 - gamma) * rep_bn


class PRepBNFullyGatedPoolFormer(FullyGatedPoolFormer):
    """Fully gated PoolFormer with all 49 LayerNorm sites under PRepBN."""

    def __init__(self, *args, repbn_bn_eps=1e-5, repbn_bn_momentum=0.1,
                 repbn_eta_init=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self._replace_layer_norms(
            self,
            bn_eps=repbn_bn_eps,
            bn_momentum=repbn_bn_momentum,
            eta_init=repbn_eta_init,
        )

    @classmethod
    def _replace_layer_norms(cls, parent, bn_eps, bn_momentum, eta_init):
        for name, child in tuple(parent.named_children()):
            if isinstance(child, LayerNorm2d):
                replacement = ProgressiveRepBatchNorm2d(
                    child.weight.numel(),
                    ln_eps=child.eps,
                    bn_eps=bn_eps,
                    bn_momentum=bn_momentum,
                    eta_init=eta_init,
                )
                with torch.no_grad():
                    replacement.ln.weight.copy_(child.weight)
                    replacement.ln.bias.copy_(child.bias)
                setattr(parent, name, replacement)
            else:
                cls._replace_layer_norms(
                    child,
                    bn_eps=bn_eps,
                    bn_momentum=bn_momentum,
                    eta_init=eta_init,
                )

    def prepbn_modules(self):
        return tuple(
            module for module in self.modules()
            if isinstance(module, ProgressiveRepBatchNorm2d)
        )

    def load_backbone_init_state_dict(self, state_dict):
        """Strictly warm-start from the accepted LayerNorm backbone.

        Original ``norm.weight``/``norm.bias`` tensors are mapped into the
        exact LayerNorm teacher.  Every original tensor must be present and no
        unknown tensor is accepted; only newly introduced RepBN state keeps its
        constructor initialization.
        """
        source = OrderedDict(state_dict)
        if source and all(key.startswith("module.") for key in source):
            source = OrderedDict(
                (key[len("module."):], value)
                for key, value in source.items())

        target = self.state_dict()
        prepbn_names = {
            name for name, module in self.named_modules()
            if isinstance(module, ProgressiveRepBatchNorm2d)
        }
        source_to_target = {}
        for target_key in target:
            matched_prepbn = next(
                (name for name in prepbn_names
                 if target_key.startswith(name + ".")),
                None,
            )
            if matched_prepbn is None:
                source_to_target[target_key] = target_key
                continue
            suffix = target_key[len(matched_prepbn) + 1:]
            if suffix in ("ln.weight", "ln.bias"):
                baseline_suffix = suffix[len("ln."):]
                source_to_target[
                    f"{matched_prepbn}.{baseline_suffix}"] = target_key

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


def poolformer_fully_gated_prepbn_s24(pretrained=False, **kwargs):
    """Warm-startable PRepBN PoolFormer-S24 with all SimpleGates active."""
    if pretrained:
        raise ValueError(
            "poolformer_fully_gated_prepbn_s24 uses backbone_init instead of "
            "the pretrained factory flag")
    model = PRepBNFullyGatedPoolFormer(
        layers=[4, 4, 12, 4],
        embed_dims=[64, 128, 320, 512],
        ffn_expands=[2.0, 2.0, 2.0, 2.0],
        downsamples=[True, True, True, True],
        layer_scale_init_value=0.0,
        **kwargs,
    )
    model.default_cfg = {}
    return model

"""Frozen-standard-deviation LayerNorm removal for fully gated PoolFormer.

This is the PoolFormer analogue of the FakeLayerNorm fine-tuning protocol in
``removing-layer-norm``. Every site starts as exact channel-wise LayerNorm and
tracks the average input standard deviation.  Sites are then switched, one at
a time, to the linear map

    (x - mean_channel(x)) / frozen_std * weight + bias.

The deployment path retains exact channel centering but contains no variance,
square root, reciprocal, or encrypted-data-dependent division.  SimpleGate is
unchanged and remains the only degree-2 nonlinearity in each block.
"""

from collections import OrderedDict

import torch
import torch.distributed as dist
import torch.nn as nn

from .poolformer_fully_gated import FullyGatedPoolFormer, LayerNorm2d


class FrozenStdLayerNorm2d(nn.Module):
    """Exact LayerNorm that can be hard-switched to a frozen-std linear map."""

    exclude_from_weight_decay = True

    def __init__(self, num_channels, eps=1e-6, momentum=0.9,
                 initial_std=1.0):
        super().__init__()
        if not 0.0 <= float(momentum) < 1.0:
            raise ValueError("momentum must be in [0, 1)")
        if float(initial_std) <= 0.0:
            raise ValueError("initial_std must be positive")

        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = float(eps)
        self.momentum = float(momentum)
        self.register_buffer("is_frozen", torch.tensor(False))
        self.register_buffer("ema_initialized", torch.tensor(False))
        self.register_buffer(
            "running_std", torch.tensor(float(initial_std), dtype=torch.float32))
        self.register_buffer(
            "frozen_std", torch.tensor(float(initial_std), dtype=torch.float32))
        self.register_buffer(
            "inverse_frozen_std",
            torch.tensor(1.0 / float(initial_std), dtype=torch.float32))
        self.register_buffer(
            "num_batches_tracked", torch.zeros((), dtype=torch.long))
        self._is_frozen = False

    def _load_from_state_dict(self, *args, **kwargs):
        super()._load_from_state_dict(*args, **kwargs)
        self._is_frozen = bool(self.is_frozen.detach().item())

    def _stable_channel_std(self, x):
        """Compute per-position channel std without squaring large values."""
        x_float = x.float()
        centered = x_float - self._stable_channel_mean(x_float)
        # Scaling by the largest centered channel keeps every squared value in
        # [0, 1].  Clamp at sqrt(eps) so the epsilon term is also safe for
        # constant and very small inputs.
        scale = centered.abs().amax(dim=1, keepdim=True).clamp_min(
            self.eps ** 0.5)
        scaled_variance = (centered / scale).square().mean(dim=1)
        scale = scale.squeeze(1)
        return scale * torch.sqrt(
            scaled_variance + self.eps / scale.square())

    @staticmethod
    def _stable_channel_mean(x):
        """Fixed-count channel mean whose accumulator cannot overflow."""
        return (x / x.shape[1]).sum(dim=1, keepdim=True)

    @torch.no_grad()
    def _update_running_std(self, x):
        position_std = self._stable_channel_std(x)
        # Avoid overflow in the reduction as well: average values after
        # normalizing by their finite maximum, then restore the scale.
        reduction_scale = position_std.amax().clamp_min(self.eps ** 0.5)
        batch_std = (
            reduction_scale * (position_std / reduction_scale).mean()
        ).to(self.running_std)
        if not torch.isfinite(batch_std) or batch_std.item() <= 0.0:
            raise FloatingPointError(
                "Non-finite frozen-std observation from finite activations")
        if (not bool(self.ema_initialized.item())
                or not bool(torch.isfinite(self.running_std).item())):
            # The second condition repairs old checkpoints whose FP32
            # centered-square statistic overflowed before this stable
            # collector was introduced.
            self.running_std.copy_(batch_std)
            self.ema_initialized.fill_(True)
        else:
            self.running_std.mul_(self.momentum).add_(
                batch_std, alpha=1.0 - self.momentum)
        self.num_batches_tracked.add_(1)

    def _exact_forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        variance = (x - mean).square().mean(dim=1, keepdim=True)
        normalized = (x - mean) * torch.rsqrt(variance + self.eps)
        return (
            self.weight.view(1, -1, 1, 1) * normalized
            + self.bias.view(1, -1, 1, 1)
        )

    def _frozen_forward(self, x):
        centered = x - self._stable_channel_mean(x)
        return (
            self.weight.view(1, -1, 1, 1)
            * centered
            * self.inverse_frozen_std.to(dtype=x.dtype)
            + self.bias.view(1, -1, 1, 1)
        )

    @torch.no_grad()
    def freeze(self, distributed=True):
        """Freeze the tracked scalar and return it.

        Each DDP rank tracks its local batches without a per-forward collective.
        The rank-local EMAs are averaged once here, at the conversion boundary.
        """
        if self._is_frozen:
            return float(self.frozen_std.item())
        if not bool(self.ema_initialized.item()):
            raise RuntimeError(
                "Cannot freeze standard deviation before observing training data")

        value = self.running_std.detach().clone()
        if (distributed and dist.is_available() and dist.is_initialized()
                and dist.get_world_size() > 1):
            dist.all_reduce(value, op=dist.ReduceOp.SUM)
            value.div_(dist.get_world_size())
        if not torch.isfinite(value) or value.item() <= 0.0:
            raise FloatingPointError(
                f"Invalid frozen standard deviation: {value.item()}")
        self.running_std.copy_(value)
        self.frozen_std.copy_(value)
        self.inverse_frozen_std.copy_(value.reciprocal())
        self.is_frozen.fill_(True)
        self._is_frozen = True
        return float(value.item())

    def forward(self, x):
        if self._is_frozen:
            return self._frozen_forward(x)
        if self.training:
            self._update_running_std(x.detach())
        return self._exact_forward(x)


class FrozenStdFullyGatedPoolFormer(FullyGatedPoolFormer):
    """Fully gated PoolFormer with 49 independently removable LayerNorm sites."""

    def __init__(self, *args, frozen_std_momentum=0.9,
                 frozen_std_initial=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self._replace_layer_norms(
            self,
            momentum=frozen_std_momentum,
            initial_std=frozen_std_initial,
        )
        self._frozen_std_groups = self._build_frozen_std_groups()
        self._frozen_std_auxiliary_enabled = False
        self._frozen_std_final_input = None

    @classmethod
    def _replace_layer_norms(cls, parent, momentum, initial_std):
        for name, child in tuple(parent.named_children()):
            if isinstance(child, LayerNorm2d):
                replacement = FrozenStdLayerNorm2d(
                    child.weight.numel(),
                    eps=child.eps,
                    momentum=momentum,
                    initial_std=initial_std,
                )
                with torch.no_grad():
                    replacement.weight.copy_(child.weight)
                    replacement.bias.copy_(child.bias)
                setattr(parent, name, replacement)
            else:
                cls._replace_layer_norms(
                    child, momentum=momentum,
                    initial_std=initial_std)

    def frozen_std_modules(self):
        return tuple(
            module for module in self.modules()
            if isinstance(module, FrozenStdLayerNorm2d)
        )

    def _blocks_in_forward_order(self):
        return tuple(
            block
            for stage in self.network
            for block in stage.children()
            if (hasattr(block, "norm1")
                and isinstance(block.norm1, FrozenStdLayerNorm2d))
        )

    def _build_frozen_std_groups(self):
        blocks = self._blocks_in_forward_order()
        groups = [(block.norm2,) for block in blocks]
        groups.extend((block.norm1,) for block in blocks)
        if not isinstance(self.norm, FrozenStdLayerNorm2d):
            raise RuntimeError("Final PoolFormer normalization was not replaced")
        groups.append((self.norm,))
        return tuple(groups)

    def frozen_std_group_names(self):
        module_names = {id(module): name for name, module in self.named_modules()}
        return tuple(
            tuple(module_names[id(module)] for module in group)
            for group in self._frozen_std_groups
        )

    def frozen_std_frozen_count(self):
        states = tuple(
            all(module._is_frozen for module in group)
            for group in self._frozen_std_groups
        )
        first_exact = next(
            (index for index, frozen in enumerate(states) if not frozen),
            len(states),
        )
        if any(states[first_exact:]):
            raise RuntimeError(
                "Frozen-std checkpoint is not a prefix of the configured schedule")
        return first_exact

    def freeze_frozen_std_group(self, group_index, distributed=True):
        group_index = int(group_index)
        if not 0 <= group_index < len(self._frozen_std_groups):
            raise IndexError(f"Invalid frozen-std group index: {group_index}")
        names = self.frozen_std_group_names()[group_index]
        values = tuple(
            module.freeze(distributed=distributed)
            for module in self._frozen_std_groups[group_index]
        )
        return tuple(zip(names, values))

    def set_frozen_std_auxiliary_loss(self, enabled):
        self._frozen_std_auxiliary_enabled = bool(enabled)
        if not self._frozen_std_auxiliary_enabled:
            self._frozen_std_final_input = None

    def frozen_std_auxiliary_loss(self):
        hidden = self._frozen_std_final_input
        self._frozen_std_final_input = None
        if hidden is None:
            raise RuntimeError(
                "Frozen-std auxiliary loss requested without a training forward")
        std = self.norm._stable_channel_std(hidden)
        target_statistics = torch.stack((
            std.detach().sum(),
            std.new_tensor(std.numel()),
        ))
        if (dist.is_available() and dist.is_initialized()
                and dist.get_world_size() > 1):
            dist.all_reduce(target_statistics, op=dist.ReduceOp.SUM)
        target = target_statistics[0] / target_statistics[1]
        return (std - target).square().mean()

    def clear_frozen_std_cached_tensors(self):
        self._frozen_std_final_input = None

    def forward(self, x):
        x = self.forward_tokens(self.forward_embeddings(x))
        if self.training and self._frozen_std_auxiliary_enabled:
            self._frozen_std_final_input = x
        else:
            self._frozen_std_final_input = None
        x = self.norm(x)
        if self.face_embedding:
            return self.head(x)
        return self.head(x.mean(dim=(-2, -1)))

    def load_backbone_init_state_dict(self, state_dict):
        """Strictly warm-start parameters from the accepted LayerNorm model."""
        source = OrderedDict(state_dict)
        if source and all(key.startswith("module.") for key in source):
            source = OrderedDict(
                (key[len("module."):], value) for key, value in source.items())

        target = self.state_dict()
        wrapper_names = {
            name for name, module in self.named_modules()
            if isinstance(module, FrozenStdLayerNorm2d)
        }
        source_to_target = {}
        for target_key in target:
            matched_name = next(
                (name for name in wrapper_names
                 if target_key.startswith(name + ".")),
                None,
            )
            if matched_name is None:
                source_to_target[target_key] = target_key
                continue
            suffix = target_key[len(matched_name) + 1:]
            if suffix in ("weight", "bias"):
                source_to_target[target_key] = target_key

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


def poolformer_fully_gated_frozen_std_s24(pretrained=False, **kwargs):
    """Warm-startable PoolFormer-S24 for hard frozen-std LN removal."""
    if pretrained:
        raise ValueError(
            "poolformer_fully_gated_frozen_std_s24 uses backbone_init instead "
            "of the pretrained factory flag")
    model = FrozenStdFullyGatedPoolFormer(
        layers=[4, 4, 12, 4],
        embed_dims=[64, 128, 320, 512],
        ffn_expands=[2.0, 2.0, 2.0, 2.0],
        downsamples=[True, True, True, True],
        layer_scale_init_value=0.0,
        **kwargs,
    )
    model.default_cfg = {}
    return model

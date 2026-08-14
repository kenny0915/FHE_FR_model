"""PoolFormer with NAFNet-style SimpleGate MLPs trained from scratch.

This backbone is a stability diagnostic, not an FHE-ready inference graph:
channel-wise LayerNorm is deliberately retained before every residual branch.
Unlike the progressive RepBatchNorm experiment, every MLP uses SimpleGate from
the first optimization step and contains no GELU teacher path.
"""

import torch
import torch.nn as nn

try:
    from timm.models.layers import DropPath, trunc_normal_
except ImportError:
    def trunc_normal_(tensor, mean=0.0, std=1.0):
        return nn.init.trunc_normal_(tensor, mean=mean, std=std)

    class DropPath(nn.Module):
        def __init__(self, drop_prob=0.0):
            super().__init__()
            self.drop_prob = float(drop_prob)

        def forward(self, x):
            if self.drop_prob == 0.0 or not self.training:
                return x
            keep_prob = 1.0 - self.drop_prob
            shape = (x.shape[0],) + (1,) * (x.ndim - 1)
            random_tensor = keep_prob + torch.rand(
                shape, dtype=x.dtype, device=x.device)
            return x.div(keep_prob) * random_tensor.floor_()


class PatchEmbed(nn.Module):
    """Convolutional patch embedding used by PoolFormer."""

    def __init__(self, patch_size, stride, padding, in_chans, embed_dim):
        super().__init__()
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size,
            stride=stride, padding=padding)

    def forward(self, x):
        return self.proj(x)


class Pooling(nn.Module):
    """PoolFormer token mixer: local average minus the identity."""

    def __init__(self, pool_size=3):
        super().__init__()
        self.pool = nn.AvgPool2d(
            pool_size, stride=1, padding=pool_size // 2,
            count_include_pad=False)

    def forward(self, x):
        return self.pool(x) - x


class LayerNorm2d(nn.Module):
    """NAFNet-style LayerNorm over channels at each spatial position."""

    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = float(eps)

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        variance = (x - mean).square().mean(dim=1, keepdim=True)
        normalized = (x - mean) * torch.rsqrt(variance + self.eps)
        return (
            self.weight.view(1, -1, 1, 1) * normalized
            + self.bias.view(1, -1, 1, 1)
        )


class SimpleGate(nn.Module):
    """Split channels into two equal operands and multiply them."""

    def forward(self, x):
        if x.shape[1] % 2 != 0:
            raise ValueError(
                f"SimpleGate needs an even channel count, got {x.shape[1]}")
        operand1, operand2 = x.chunk(2, dim=1)
        return operand1 * operand2


class GatedMlp(nn.Module):
    """NAFNet-width feed-forward branch: C -> 2C -> gate -> C -> C."""

    def __init__(self, in_features, pre_gate_features=None,
                 out_features=None, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        pre_gate_features = pre_gate_features or 2 * in_features
        if pre_gate_features % 2 != 0:
            raise ValueError(
                "pre_gate_features must be even for SimpleGate, got "
                f"{pre_gate_features}")

        self.fc1 = nn.Conv2d(in_features, pre_gate_features, 1)
        self.act = SimpleGate()
        self.fc2 = nn.Conv2d(pre_gate_features // 2, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Conv2d):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return self.drop(x)


class PoolFormerBlock(nn.Module):
    """PoolFormer token mixer plus a fully gated NAFNet-style MLP."""

    def __init__(self, dim, pool_size=3, ffn_expand=2.0, drop=0.0,
                 drop_path=0.0, layer_scale_init_value=0.0):
        super().__init__()
        pre_gate_features = int(dim * ffn_expand)
        self.norm1 = LayerNorm2d(dim)
        self.token_mixer = Pooling(pool_size=pool_size)
        self.norm2 = LayerNorm2d(dim)
        self.mlp = GatedMlp(
            in_features=dim,
            pre_gate_features=pre_gate_features,
            out_features=dim,
            drop=drop,
        )
        self.drop_path = (
            DropPath(drop_path) if drop_path > 0.0 else nn.Identity())
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones(dim))
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones(dim))

    @staticmethod
    def _scale(scale, value):
        return scale.view(1, -1, 1, 1) * value

    def forward(self, x):
        x = x + self.drop_path(self._scale(
            self.layer_scale_1, self.token_mixer(self.norm1(x))))
        x = x + self.drop_path(self._scale(
            self.layer_scale_2, self.mlp(self.norm2(x))))
        return x


def _basic_blocks(dim, stage_index, layers, pool_size=3, ffn_expand=2.0,
                  drop_rate=0.0, drop_path_rate=0.0,
                  layer_scale_init_value=0.0):
    blocks = []
    total_blocks = sum(layers)
    for block_index in range(layers[stage_index]):
        global_index = block_index + sum(layers[:stage_index])
        block_drop_path = (
            drop_path_rate * global_index / max(total_blocks - 1, 1))
        blocks.append(PoolFormerBlock(
            dim=dim,
            pool_size=pool_size,
            ffn_expand=ffn_expand,
            drop=drop_rate,
            drop_path=block_drop_path,
            layer_scale_init_value=layer_scale_init_value,
        ))
    return nn.Sequential(*blocks)


class FullyGatedPoolFormer(nn.Module):
    """Face-recognition PoolFormer whose MLPs are gated from initialization."""

    def __init__(self, layers, embed_dims, ffn_expands, downsamples,
                 pool_size=3, num_classes=512,
                 in_patch_size=3, in_stride=2, in_pad=1,
                 down_patch_size=3, down_stride=2, down_pad=1,
                 drop_rate=0.0, drop_path_rate=0.0,
                 layer_scale_init_value=0.0, face_embedding=True,
                 fp16=False, **kwargs):
        super().__init__()
        if fp16:
            raise ValueError(
                "FullyGatedPoolFormer is an FP32 stability experiment; "
                "set fp16=False")
        if not (len(layers) == len(embed_dims) == len(ffn_expands)
                == len(downsamples)):
            raise ValueError(
                "layers, embed_dims, ffn_expands, and downsamples must have "
                "the same length")

        self.num_classes = num_classes
        self.face_embedding = bool(face_embedding)
        self.fp16 = False
        self.patch_embed = PatchEmbed(
            patch_size=in_patch_size,
            stride=in_stride,
            padding=in_pad,
            in_chans=3,
            embed_dim=embed_dims[0],
        )

        network = []
        for stage_index, dim in enumerate(embed_dims):
            network.append(_basic_blocks(
                dim=dim,
                stage_index=stage_index,
                layers=layers,
                pool_size=pool_size,
                ffn_expand=ffn_expands[stage_index],
                drop_rate=drop_rate,
                drop_path_rate=drop_path_rate,
                layer_scale_init_value=layer_scale_init_value,
            ))
            if stage_index == len(layers) - 1:
                break
            if (downsamples[stage_index]
                    or embed_dims[stage_index] != embed_dims[stage_index + 1]):
                network.append(PatchEmbed(
                    patch_size=down_patch_size,
                    stride=down_stride,
                    padding=down_pad,
                    in_chans=dim,
                    embed_dim=embed_dims[stage_index + 1],
                ))
        self.network = nn.ModuleList(network)
        self.norm = LayerNorm2d(embed_dims[-1])

        if self.face_embedding:
            self.head = nn.Sequential(
                nn.Conv2d(
                    embed_dims[-1], embed_dims[-1], kernel_size=7,
                    stride=1, padding=0),
                nn.BatchNorm2d(embed_dims[-1]),
                nn.Flatten(),
                nn.Linear(embed_dims[-1], num_classes, bias=False),
                nn.BatchNorm1d(num_classes),
            )
        else:
            self.head = (
                nn.Linear(embed_dims[-1], num_classes)
                if num_classes > 0 else nn.Identity())
        self.apply(self._init_linear_weights)

    @staticmethod
    def _init_linear_weights(module):
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward_embeddings(self, x):
        return self.patch_embed(x)

    def forward_tokens(self, x):
        for block in self.network:
            x = block(x)
        return x

    def forward(self, x):
        x = self.forward_tokens(self.forward_embeddings(x))
        x = self.norm(x)
        if self.face_embedding:
            return self.head(x)
        return self.head(x.mean(dim=(-2, -1)))


def poolformer_fully_gated_s24(pretrained=False, **kwargs):
    """PoolFormer-S24 with exact NAFNet FFN expansion and all gates active."""
    if pretrained:
        raise ValueError(
            "poolformer_fully_gated_s24 must be trained from scratch")
    model = FullyGatedPoolFormer(
        layers=[4, 4, 12, 4],
        embed_dims=[64, 128, 320, 512],
        ffn_expands=[2.0, 2.0, 2.0, 2.0],
        downsamples=[True, True, True, True],
        layer_scale_init_value=0.0,
        **kwargs,
    )
    model.default_cfg = {}
    return model

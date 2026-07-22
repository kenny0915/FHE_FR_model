# Copyright 2021 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
PoolFormer with RepBatchNorm normalization and SimpleGate activation.
"""
import copy
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import poolformer_no_ln as base


BatchNorm2d = base.BatchNorm2d
DropPath = base.DropPath
PatchEmbed = base.PatchEmbed
Pooling = base.Pooling
default_cfgs = base.default_cfgs
model_urls = base.model_urls
register_model = base.register_model
trunc_normal_ = base.trunc_normal_


class SimpleGate(nn.Module):
    """Progressive GELU-to-SimpleGate activation with instrumentation.

    This is a bilinear gate ``x1 * x2``, not the scalar activation ``x**2``.
    A fixed-width teacher path makes it possible to start from ``GELU(x1)``
    and blend into the gate without changing either convolution's shape.
    Statistics are captured only when explicitly enabled so normal training has
    no quantile/reduction overhead. The multiplication can be evaluated in
    float32 under AMP to avoid an avoidable fp16 overflow during training.
    """
    def __init__(self, range_limit=6.0, sample_size=16384,
                 compute_fp32=True, fail_on_nonfinite=True,
                 conversion_group=0, initial_blend=0.0):
        super().__init__()
        if range_limit <= 0:
            raise ValueError("SimpleGate range_limit must be positive")
        if sample_size <= 0:
            raise ValueError("SimpleGate sample_size must be positive")
        self.range_limit = float(range_limit)
        self.sample_size = int(sample_size)
        self.compute_fp32 = bool(compute_fp32)
        self.fail_on_nonfinite = bool(fail_on_nonfinite)
        self.conversion_group = int(conversion_group)
        # This is scheduled state rather than a learned value. Keeping it as a
        # Python scalar avoids 24 GPU synchronizations in every forward pass;
        # the training schedule reconstructs it after loading a checkpoint.
        self.blend = float(initial_blend)
        self.instrumentation_enabled = False
        self.auxiliary_losses_enabled = False
        self.gradient_scale = 1.0
        self._last_stats = None
        self._last_teacher = None
        self._last_product = None
        self._last_operand1 = None
        self._last_operand2 = None

    def set_blend(self, blend):
        blend = float(blend)
        if not 0.0 <= blend <= 1.0:
            raise ValueError(f"SimpleGate blend must be in [0, 1], got {blend}")
        self.blend = blend

    def distillation_loss(self):
        if self._last_product is None or self._last_teacher is None:
            return None
        return F.mse_loss(self._last_product, self._last_teacher.detach())

    def range_penalty(self):
        if self._last_operand1 is None or self._last_operand2 is None:
            return None
        limit = self.range_limit
        return 0.5 * (
            F.relu(self._last_operand1.abs() - limit).square().mean()
            + F.relu(self._last_operand2.abs() - limit).square().mean()
        )

    def set_instrumentation(self, enabled=True, gradient_scale=1.0):
        self.instrumentation_enabled = bool(enabled)
        self.gradient_scale = max(float(gradient_scale), 1.0)
        if not enabled:
            self._last_stats = None

    def set_auxiliary_losses(self, enabled=True):
        self.auxiliary_losses_enabled = bool(enabled)
        if not enabled:
            self._last_teacher = None
            self._last_product = None
            self._last_operand1 = None
            self._last_operand2 = None

    def range_stats(self):
        return self._last_stats

    def _sample(self, tensor):
        flat = tensor.detach().flatten()
        if flat.numel() <= self.sample_size:
            return flat
        stride = (flat.numel() + self.sample_size - 1) // self.sample_size
        return flat[::stride][:self.sample_size]

    def _sample_for_loss(self, tensor):
        """Bound auxiliary-loss memory while retaining sampled gradients."""
        flat = tensor.flatten()
        if flat.numel() <= self.sample_size:
            return flat
        stride = (flat.numel() + self.sample_size - 1) // self.sample_size
        return flat[::stride][:self.sample_size]

    @staticmethod
    def _rms(tensor):
        return tensor.square().mean().sqrt()

    @staticmethod
    def _absmax(tensor):
        detached = tensor.detach()
        return torch.maximum(detached.amin().abs(), detached.amax().abs()).float()

    def _capture_forward_stats(self, x1, x2, product):
        with torch.no_grad():
            # Sample before converting to float32. Converting the complete
            # stage-1 activation at batch size 256 would retain hundreds of MB
            # merely for instrumentation.
            sample1 = self._sample(x1).float()
            sample2 = self._sample(x2).float()
            sample_product = self._sample(product).float()
            finite = (
                torch.isfinite(sample1).all()
                & torch.isfinite(sample2).all()
                & torch.isfinite(sample_product).all()
            )
            if self.fail_on_nonfinite and not bool(finite.item()):
                raise FloatingPointError("Non-finite value in SimpleGate operands or output")

            mean1 = sample1.mean()
            mean2 = sample2.mean()
            centered1 = sample1 - mean1
            centered2 = sample2 - mean2
            correlation = (centered1 * centered2).mean() / (
                self._rms(centered1) * self._rms(centered2) + 1e-12)
            limit = self.range_limit
            self._last_stats = {
                # Maxima are exact full-tensor reductions; percentile, RMS,
                # correlation, and outside fractions use the bounded sample.
                "operand1_absmax": self._absmax(x1),
                "operand2_absmax": self._absmax(x2),
                "product_absmax": self._absmax(product),
                "operand1_rms": self._rms(sample1),
                "operand2_rms": self._rms(sample2),
                "product_rms": self._rms(sample_product),
                "operand1_p999": torch.quantile(sample1.abs(), 0.999),
                "operand2_p999": torch.quantile(sample2.abs(), 0.999),
                "product_p999": torch.quantile(sample_product.abs(), 0.999),
                "operand1_outside_fraction": (sample1.abs() > limit).float().mean(),
                "operand2_outside_fraction": (sample2.abs() > limit).float().mean(),
                "product_outside_fraction": (
                    (sample_product.abs() > limit).float().mean()),
                "operand_correlation": correlation,
                "finite": finite.float(),
            }

    def _capture_gradient_stats(self, grad):
        with torch.no_grad():
            sample = self._sample(grad).float() / self.gradient_scale
            finite = torch.isfinite(sample).all()
            if self.fail_on_nonfinite and not bool(finite.item()):
                raise FloatingPointError("Non-finite SimpleGate output gradient")
            if self._last_stats is not None:
                self._last_stats.update({
                    "gradient_absmax": sample.abs().amax(),
                    "gradient_rms": self._rms(sample),
                    "gradient_finite": finite.float(),
                })
        return grad

    def forward(self, x):
        if x.shape[1] % 2 != 0:
            raise ValueError(
                f"SimpleGate needs an even channel count, got {x.shape[1]}")
        x1, x2 = x.chunk(2, dim=1)
        blend = self.blend
        need_full_product = blend > 0.0 or self.instrumentation_enabled
        if need_full_product:
            if self.compute_fp32 and x.dtype in (
                    torch.float16, torch.bfloat16):
                # Keep the result in fp32. Casting a finite large product back
                # to fp16 here can itself create inf before the next autocast op.
                product = x1.float() * x2.float()
            else:
                product = x1 * x2
        else:
            product = None

        need_teacher = blend < 1.0 or (
            self.training and self.auxiliary_losses_enabled)
        teacher = F.gelu(x1) if need_teacher else None
        if blend <= 0.0:
            output = teacher
        elif blend >= 1.0:
            output = product
        else:
            output = torch.lerp(teacher.float(), product.float(), blend)

        if self.training and self.auxiliary_losses_enabled:
            # The auxiliary losses warm the multiplier branch before it enters
            # the main path and keep it close to GELU throughout conversion.
            loss_operand1 = self._sample_for_loss(x1)
            loss_operand2 = self._sample_for_loss(x2)
            self._last_teacher = F.gelu(loss_operand1).detach()
            self._last_product = (
                loss_operand1.float() * loss_operand2.float())
            self._last_operand1 = loss_operand1
            self._last_operand2 = loss_operand2
        else:
            self._last_teacher = None
            self._last_product = None
            self._last_operand1 = None
            self._last_operand2 = None

        if self.instrumentation_enabled:
            self._capture_forward_stats(x1, x2, product)
            self._last_stats["blend"] = product.new_tensor(self.blend).float()
            if teacher is not None:
                teacher_error = product.float() - teacher.detach().float()
                self._last_stats["teacher_error_rms"] = self._rms(
                    self._sample(teacher_error))
            if output.requires_grad:
                output.register_hook(self._capture_gradient_stats)
        return output


class Mlp(nn.Module):
    """
    Implementation of MLP with 1*1 convolutions and SimpleGate.
    Input: tensor with shape [B, C, H, W]
    """
    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=SimpleGate, drop=0.,
                 gate_range_limit=6.0, gate_stats_sample_size=16384,
                 gate_compute_fp32=True, gate_fail_on_nonfinite=True,
                 gate_conversion_group=0, gate_initial_blend=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features * 2, 1)
        if act_layer is SimpleGate:
            self.act = act_layer(
                range_limit=gate_range_limit,
                sample_size=gate_stats_sample_size,
                compute_fp32=gate_compute_fp32,
                fail_on_nonfinite=gate_fail_on_nonfinite,
                conversion_group=gate_conversion_group,
                initial_blend=gate_initial_blend,
            )
        else:
            self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)
        if act_layer is SimpleGate:
            self._init_gate_multiplier()

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _init_gate_multiplier(self):
        """Initialize x2 so x1*x2 matches GELU near the origin.

        GELU(x) = x Phi(x) and Phi(x) is locally
        ``0.5 + x / sqrt(2*pi)``.
        """
        half = self.fc1.out_channels // 2
        inverse_sqrt_2pi = 0.3989422804014327
        with torch.no_grad():
            self.fc1.weight[half:].copy_(
                inverse_sqrt_2pi * self.fc1.weight[:half])
            if self.fc1.bias is not None:
                self.fc1.bias[half:].copy_(
                    0.5 + inverse_sqrt_2pi * self.fc1.bias[:half])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class PoolFormerBlock(nn.Module):
    """
    Implementation of one PoolFormer block.
    --dim: embedding dim
    --pool_size: pooling size
    --mlp_ratio: mlp expansion ratio
    --act_layer: activation
    --norm_layer: normalization
    --drop: dropout rate
    --drop path: Stochastic Depth,
        refer to https://arxiv.org/abs/1603.09382
    --use_layer_scale, --layer_scale_init_value: LayerScale,
        refer to https://arxiv.org/abs/2103.17239
    """
    def __init__(self, dim, pool_size=3, mlp_ratio=4.,
                 act_layer=SimpleGate, norm_layer=BatchNorm2d,
                 drop=0., drop_path=0.,
                 use_layer_scale=True, layer_scale_init_value=1e-5,
                 gate_range_limit=6.0, gate_stats_sample_size=16384,
                 gate_compute_fp32=True, gate_fail_on_nonfinite=True,
                 gate_conversion_group=0, gate_initial_blend=0.0):

        super().__init__()

        self.norm1 = norm_layer(dim)
        self.token_mixer = Pooling(pool_size=pool_size)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop,
                       gate_range_limit=gate_range_limit,
                       gate_stats_sample_size=gate_stats_sample_size,
                       gate_compute_fp32=gate_compute_fp32,
                       gate_fail_on_nonfinite=gate_fail_on_nonfinite,
                       gate_conversion_group=gate_conversion_group,
                       gate_initial_blend=gate_initial_blend)

        # The following two techniques are useful to train deep PoolFormers.
        self.drop_path = DropPath(drop_path) if drop_path > 0. \
            else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(
                layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(
                layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        if self.use_layer_scale:
            x = x + self.drop_path(
                self.layer_scale_1.unsqueeze(-1).unsqueeze(-1)
                * self.token_mixer(self.norm1(x)))
            x = x + self.drop_path(
                self.layer_scale_2.unsqueeze(-1).unsqueeze(-1)
                * self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.token_mixer(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


def basic_blocks(dim, index, layers,
                 pool_size=3, mlp_ratio=4.,
                 act_layer=SimpleGate, norm_layer=BatchNorm2d,
                 drop_rate=.0, drop_path_rate=0.,
                 use_layer_scale=True, layer_scale_init_value=1e-5,
                 gate_range_limit=6.0, gate_stats_sample_size=16384,
                 gate_compute_fp32=True, gate_fail_on_nonfinite=True,
                 gate_initial_blend=0.0):
    """
    generate PoolFormer blocks for a stage
    return: PoolFormer blocks
    """
    blocks = []
    for block_idx in range(layers[index]):
        if index < 2:
            conversion_group = index
        elif index == 2:
            # Divide the longest stage into three contiguous groups. For S24
            # these are blocks 0-3, 4-7, and 8-11.
            conversion_group = 2 + min(
                2, (3 * block_idx) // layers[index])
        else:
            conversion_group = 5
        block_dpr = drop_path_rate * (
            block_idx + sum(layers[:index])) / (sum(layers) - 1)
        blocks.append(PoolFormerBlock(
            dim, pool_size=pool_size, mlp_ratio=mlp_ratio,
            act_layer=act_layer, norm_layer=norm_layer,
            drop=drop_rate, drop_path=block_dpr,
            use_layer_scale=use_layer_scale,
            layer_scale_init_value=layer_scale_init_value,
            gate_range_limit=gate_range_limit,
            gate_stats_sample_size=gate_stats_sample_size,
            gate_compute_fp32=gate_compute_fp32,
            gate_fail_on_nonfinite=gate_fail_on_nonfinite,
            gate_conversion_group=conversion_group,
            gate_initial_blend=gate_initial_blend,
            ))
    blocks = nn.Sequential(*blocks)

    return blocks


class PoolFormer(nn.Module):
    """
    PoolFormer, the main class of our model
    --layers: [x,x,x,x], number of blocks for the 4 stages
    --embed_dims, --mlp_ratios, --pool_size: the embedding dims, mlp ratios and
        pooling size for the 4 stages
    --downsamples: flags to apply downsampling or not
    --norm_layer, --act_layer: define the types of normalization and activation
    --num_classes: number of classes for the image classification
    --in_patch_size, --in_stride, --in_pad: specify the patch embedding
        for the input image
    --down_patch_size --down_stride --down_pad:
        specify the downsample (patch embed.)
    --fork_feat: whether output features of the 4 stages, for dense prediction
    --init_cfg, --pretrained:
        for mmdetection and mmsegmentation to load pretrained weights
    """
    def __init__(self, layers, embed_dims=None,
                 mlp_ratios=None, downsamples=None,
                 pool_size=3,
                 norm_layer=BatchNorm2d, act_layer=SimpleGate,
                 num_classes=512,
                 in_patch_size=3, in_stride=2, in_pad=1,
                 down_patch_size=3, down_stride=2, down_pad=1,
                 drop_rate=0., drop_path_rate=0.,
                 use_layer_scale=True, layer_scale_init_value=1e-5,
                 gate_range_limit=6.0,
                 gate_stats_sample_size=16384,
                 gate_compute_fp32=True,
                 gate_fail_on_nonfinite=True,
                 gate_initial_blend=0.0,
                 fork_feat=False,
                 face_embedding=True,
                 fp16=False,
                 init_cfg=None,
                 pretrained=None,
                 **kwargs):

        super().__init__()

        if not fork_feat:
            self.num_classes = num_classes
        self.fork_feat = fork_feat
        self.face_embedding = face_embedding
        self.fp16 = fp16

        self.patch_embed = PatchEmbed(
            patch_size=in_patch_size, stride=in_stride, padding=in_pad,
            in_chans=3, embed_dim=embed_dims[0])

        # set the main block in network
        network = []
        for i in range(len(layers)):
            stage = basic_blocks(embed_dims[i], i, layers,
                                 pool_size=pool_size, mlp_ratio=mlp_ratios[i],
                                 act_layer=act_layer, norm_layer=norm_layer,
                                 drop_rate=drop_rate,
                                 drop_path_rate=drop_path_rate,
                                 use_layer_scale=use_layer_scale,
                                 layer_scale_init_value=layer_scale_init_value,
                                 gate_range_limit=gate_range_limit,
                                 gate_stats_sample_size=gate_stats_sample_size,
                                 gate_compute_fp32=gate_compute_fp32,
                                 gate_fail_on_nonfinite=gate_fail_on_nonfinite,
                                 gate_initial_blend=gate_initial_blend)
            network.append(stage)
            if i >= len(layers) - 1:
                break
            if downsamples[i] or embed_dims[i] != embed_dims[i+1]:
                # downsampling between two stages
                network.append(
                    PatchEmbed(
                        patch_size=down_patch_size, stride=down_stride,
                        padding=down_pad,
                        in_chans=embed_dims[i], embed_dim=embed_dims[i+1]
                        )
                    )

        self.network = nn.ModuleList(network)

        if self.fork_feat:
            # add a norm layer for each output
            self.out_indices = [0, 2, 4, 6]
            for i_emb, i_layer in enumerate(self.out_indices):
                if i_emb == 0 and os.environ.get('FORK_LAST3', None):
                    layer = nn.Identity()
                else:
                    layer = norm_layer(embed_dims[i_emb])
                layer_name = f'norm{i_layer}'
                self.add_module(layer_name, layer)
        else:
            self.norm = norm_layer(embed_dims[-1])
            # modify the head for face recognition, which is a conv layer followed by a linear layer
            if face_embedding:
                self.head = nn.Sequential(
                    nn.Conv2d(embed_dims[-1], embed_dims[-1], kernel_size=(7, 7), stride=(1, 1), padding=(0, 0), groups=1),
                    nn.BatchNorm2d(num_features=embed_dims[-1]),
                    nn.Flatten(),
                    nn.Linear(embed_dims[-1], num_classes, bias=False),
                    nn.BatchNorm1d(num_classes))

            else:
                # Classifier head
                self.head = nn.Linear(
                    embed_dims[-1], num_classes) if num_classes > 0 \
                    else nn.Identity()

        self.apply(self.cls_init_weights)

        self.init_cfg = copy.deepcopy(init_cfg)
        # load pre-trained model
        if self.fork_feat and (
                self.init_cfg is not None or pretrained is not None):
            self.init_weights()

    # init for classification
    def cls_init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    # init for mmdetection or mmsegmentation by loading
    # imagenet pre-trained weights
    def init_weights(self, pretrained=None):
        logger = base.get_root_logger()
        if self.init_cfg is None and pretrained is None:
            logger.warn(f'No pre-trained weights for '
                        f'{self.__class__.__name__}, '
                        f'training start from scratch')
            pass
        else:
            assert 'checkpoint' in self.init_cfg, f'Only support ' \
                                                  f'specify `Pretrained` in ' \
                                                  f'`init_cfg` in ' \
                                                  f'{self.__class__.__name__} '
            if self.init_cfg is not None:
                ckpt_path = self.init_cfg['checkpoint']
            elif pretrained is not None:
                ckpt_path = pretrained

            ckpt = base._load_checkpoint(
                ckpt_path, logger=logger, map_location='cpu')
            if 'state_dict' in ckpt:
                _state_dict = ckpt['state_dict']
            elif 'model' in ckpt:
                _state_dict = ckpt['model']
            else:
                _state_dict = ckpt

            state_dict = _state_dict
            missing_keys, unexpected_keys = \
                self.load_state_dict(state_dict, False)

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes):
        self.num_classes = num_classes
        self.head = nn.Linear(
            self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_embeddings(self, x):
        x = self.patch_embed(x)
        return x

    def forward_tokens(self, x):
        outs = []
        for idx, block in enumerate(self.network):
            x = block(x)
            if self.fork_feat and idx in self.out_indices:
                norm_layer = getattr(self, f'norm{idx}')
                x_out = norm_layer(x)
                outs.append(x_out)
        if self.fork_feat:
            # output the features of four stages for dense prediction
            return outs
        # output only the features of last layer for image classification
        return x

    def forward(self, x):
        with torch.cuda.amp.autocast(self.fp16):
            # input embedding
            x = self.forward_embeddings(x)
            # through backbone
            x = self.forward_tokens(x)
        if self.fork_feat:
            # output features of four stages for dense prediction
            return x
        x = self.norm(x)
        if self.face_embedding:
            return self.head(x.float() if self.fp16 else x)
        x = x.mean([-2, -1])
        cls_out = self.head(x)
        # for image classification
        return cls_out

    def simple_gates(self):
        return {
            name: module for name, module in self.named_modules()
            if isinstance(module, SimpleGate)
        }

    def set_simple_gate_instrumentation(self, enabled=True, gradient_scale=1.0):
        for gate in self.simple_gates().values():
            gate.set_instrumentation(enabled, gradient_scale=gradient_scale)

    def set_simple_gate_auxiliary_losses(self, enabled=True):
        for gate in self.simple_gates().values():
            gate.set_auxiliary_losses(enabled)

    def set_simple_gate_blends(self, blends):
        blends = tuple(float(value) for value in blends)
        expected_groups = 1 + max(
            (gate.conversion_group for gate in self.simple_gates().values()),
            default=-1,
        )
        if len(blends) != expected_groups:
            raise ValueError(
                f"Expected {expected_groups} SimpleGate blends, got {len(blends)}")
        for gate in self.simple_gates().values():
            gate.set_blend(blends[gate.conversion_group])

    def simple_gate_group_names(self):
        groups = {}
        for name, gate in self.simple_gates().items():
            groups.setdefault(gate.conversion_group, []).append(name)
        return tuple(tuple(groups[index]) for index in sorted(groups))

    def simple_gate_distillation_loss(self):
        losses = [
            gate.distillation_loss() for gate in self.simple_gates().values()
        ]
        losses = [loss for loss in losses if loss is not None]
        if losses:
            return torch.stack(losses).mean()
        return next(self.parameters()).new_zeros(())

    def simple_gate_range_penalty(self):
        penalties = [
            gate.range_penalty() for gate in self.simple_gates().values()
        ]
        penalties = [penalty for penalty in penalties if penalty is not None]
        if penalties:
            return torch.stack(penalties).mean()
        return next(self.parameters()).new_zeros(())

    def simple_gate_range_stats(self):
        modules = dict(self.named_modules())
        stats = {}
        for name, gate in self.simple_gates().items():
            gate_stats = gate.range_stats()
            if gate_stats is None:
                continue
            layer_stats = dict(gate_stats)
            block_name = name.rsplit(".mlp.act", 1)[0]
            block = modules.get(block_name)
            if block is not None and getattr(block, "use_layer_scale", False):
                scale = block.layer_scale_2.detach().float()
                layer_stats.update({
                    "residual_scale_absmax": scale.abs().amax(),
                    "residual_scale_rms": scale.square().mean().sqrt(),
                })
            stats[name] = layer_stats
        return stats


def _load_pretrained_if_requested(model, pretrained, model_name):
    if pretrained:
        url = model_urls[model_name]
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint)


@register_model
def poolformer_s12(pretrained=False, **kwargs):
    """
    PoolFormer-S12 model, Params: 12M
    --layers: [x,x,x,x], numbers of layers for the four stages
    --embed_dims, --mlp_ratios:
        embedding dims and mlp ratios for the four stages
    --downsamples: flags to apply downsampling or not in four blocks
    """
    layers = [2, 2, 6, 2]
    embed_dims = [64, 128, 320, 512]
    mlp_ratios = [4, 4, 4, 4]
    downsamples = [True, True, True, True]
    model = PoolFormer(
        layers, embed_dims=embed_dims,
        mlp_ratios=mlp_ratios, downsamples=downsamples,
        **kwargs)
    model.default_cfg = default_cfgs['poolformer_s']
    _load_pretrained_if_requested(model, pretrained, 'poolformer_s12')
    return model


@register_model
def poolformer_s24(pretrained=False, **kwargs):
    """
    PoolFormer-S24 model, Params: 21M
    """
    layers = [4, 4, 12, 4]
    embed_dims = [64, 128, 320, 512]
    mlp_ratios = [4, 4, 4, 4]
    downsamples = [True, True, True, True]
    model = PoolFormer(
        layers, embed_dims=embed_dims,
        mlp_ratios=mlp_ratios, downsamples=downsamples,
        **kwargs)
    model.default_cfg = default_cfgs['poolformer_s']
    _load_pretrained_if_requested(model, pretrained, 'poolformer_s24')
    return model


@register_model
def poolformer_s24_mlp2(pretrained=False, **kwargs):
    """
    FHE-oriented PoolFormer-S24 with NAFNet SimpleGate and MLP ratio 2.

    Zero-initialized LayerScale is the PoolFormer equivalent of the paper's
    skip-init: every residual branch starts as an identity mapping and learns
    its contribution before large gate products can perturb the main path.
    """
    layers = [4, 4, 12, 4]
    embed_dims = [64, 128, 320, 512]
    mlp_ratios = [2, 2, 2, 2]
    downsamples = [True, True, True, True]
    kwargs.setdefault("layer_scale_init_value", 0.0)
    model = PoolFormer(
        layers, embed_dims=embed_dims,
        mlp_ratios=mlp_ratios, downsamples=downsamples,
        **kwargs)
    model.default_cfg = default_cfgs['poolformer_s']
    return model


@register_model
def poolformer_s36(pretrained=False, **kwargs):
    """
    PoolFormer-S36 model, Params: 31M
    """
    layers = [6, 6, 18, 6]
    embed_dims = [64, 128, 320, 512]
    mlp_ratios = [4, 4, 4, 4]
    downsamples = [True, True, True, True]
    model = PoolFormer(
        layers, embed_dims=embed_dims,
        mlp_ratios=mlp_ratios, downsamples=downsamples,
        layer_scale_init_value=1e-6,
        **kwargs)
    model.default_cfg = default_cfgs['poolformer_s']
    _load_pretrained_if_requested(model, pretrained, 'poolformer_s36')
    return model


@register_model
def poolformer_m36(pretrained=False, **kwargs):
    """
    PoolFormer-M36 model, Params: 56M
    """
    layers = [6, 6, 18, 6]
    embed_dims = [96, 192, 384, 768]
    mlp_ratios = [4, 4, 4, 4]
    downsamples = [True, True, True, True]
    model = PoolFormer(
        layers, embed_dims=embed_dims,
        mlp_ratios=mlp_ratios, downsamples=downsamples,
        layer_scale_init_value=1e-6,
        **kwargs)
    model.default_cfg = default_cfgs['poolformer_m']
    _load_pretrained_if_requested(model, pretrained, 'poolformer_m36')
    return model


@register_model
def poolformer_m48(pretrained=False, **kwargs):
    """
    PoolFormer-M48 model, Params: 73M
    """
    layers = [8, 8, 24, 8]
    embed_dims = [96, 192, 384, 768]
    mlp_ratios = [4, 4, 4, 4]
    downsamples = [True, True, True, True]
    model = PoolFormer(
        layers, embed_dims=embed_dims,
        mlp_ratios=mlp_ratios, downsamples=downsamples,
        layer_scale_init_value=1e-6,
        **kwargs)
    model.default_cfg = default_cfgs['poolformer_m']
    _load_pretrained_if_requested(model, pretrained, 'poolformer_m48')
    return model


if base.has_mmseg and base.has_mmdet:
    """
    The following models are for dense prediction based on
    mmdetection and mmsegmentation
    """
    @base.seg_BACKBONES.register_module()
    @base.det_BACKBONES.register_module()
    class simple_gate_poolformer_s12_feat(PoolFormer):
        """
        PoolFormer-S12 model, Params: 12M
        """
        def __init__(self, **kwargs):
            layers = [2, 2, 6, 2]
            embed_dims = [64, 128, 320, 512]
            mlp_ratios = [4, 4, 4, 4]
            downsamples = [True, True, True, True]
            super().__init__(
                layers, embed_dims=embed_dims,
                mlp_ratios=mlp_ratios, downsamples=downsamples,
                fork_feat=True,
                **kwargs)

    @base.seg_BACKBONES.register_module()
    @base.det_BACKBONES.register_module()
    class simple_gate_poolformer_s24_feat(PoolFormer):
        """
        PoolFormer-S24 model, Params: 21M
        """
        def __init__(self, **kwargs):
            layers = [4, 4, 12, 4]
            embed_dims = [64, 128, 320, 512]
            mlp_ratios = [4, 4, 4, 4]
            downsamples = [True, True, True, True]
            super().__init__(
                layers, embed_dims=embed_dims,
                mlp_ratios=mlp_ratios, downsamples=downsamples,
                fork_feat=True,
                **kwargs)

    @base.seg_BACKBONES.register_module()
    @base.det_BACKBONES.register_module()
    class simple_gate_poolformer_s36_feat(PoolFormer):
        """
        PoolFormer-S36 model, Params: 31M
        """
        def __init__(self, **kwargs):
            layers = [6, 6, 18, 6]
            embed_dims = [64, 128, 320, 512]
            mlp_ratios = [4, 4, 4, 4]
            downsamples = [True, True, True, True]
            super().__init__(
                layers, embed_dims=embed_dims,
                mlp_ratios=mlp_ratios, downsamples=downsamples,
                layer_scale_init_value=1e-6,
                fork_feat=True,
                **kwargs)

    @base.seg_BACKBONES.register_module()
    @base.det_BACKBONES.register_module()
    class simple_gate_poolformer_m36_feat(PoolFormer):
        """
        PoolFormer-S36 model, Params: 56M
        """
        def __init__(self, **kwargs):
            layers = [6, 6, 18, 6]
            embed_dims = [96, 192, 384, 768]
            mlp_ratios = [4, 4, 4, 4]
            downsamples = [True, True, True, True]
            super().__init__(
                layers, embed_dims=embed_dims,
                mlp_ratios=mlp_ratios, downsamples=downsamples,
                layer_scale_init_value=1e-6,
                fork_feat=True,
                **kwargs)

    @base.seg_BACKBONES.register_module()
    @base.det_BACKBONES.register_module()
    class simple_gate_poolformer_m48_feat(PoolFormer):
        """
        PoolFormer-M48 model, Params: 73M
        """
        def __init__(self, **kwargs):
            layers = [8, 8, 24, 8]
            embed_dims = [96, 192, 384, 768]
            mlp_ratios = [4, 4, 4, 4]
            downsamples = [True, True, True, True]
            super().__init__(
                layers, embed_dims=embed_dims,
                mlp_ratios=mlp_ratios, downsamples=downsamples,
                layer_scale_init_value=1e-6,
                fork_feat=True,
                **kwargs)

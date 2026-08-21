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
PoolFormer implementation
"""
import os
import copy
import torch
import torch.nn as nn

try:
    from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
    from timm.models.layers import DropPath, trunc_normal_
    from timm.models.registry import register_model
    from timm.models.layers.helpers import to_2tuple
except ImportError:
    IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

    def to_2tuple(x):
        return x if isinstance(x, tuple) else (x, x)

    def register_model(fn):
        return fn

    def trunc_normal_(tensor, mean=0., std=1.):
        return nn.init.trunc_normal_(tensor, mean=mean, std=std)

    class DropPath(nn.Module):
        def __init__(self, drop_prob=0.):
            super().__init__()
            self.drop_prob = drop_prob

        def forward(self, x):
            if self.drop_prob == 0. or not self.training:
                return x
            keep_prob = 1 - self.drop_prob
            shape = (x.shape[0],) + (1,) * (x.ndim - 1)
            random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
            random_tensor.floor_()
            return x.div(keep_prob) * random_tensor


try:
    from mmseg.models.builder import BACKBONES as seg_BACKBONES
    from mmseg.utils import get_root_logger
    from mmcv.runner import _load_checkpoint
    has_mmseg = True
except ImportError:
    has_mmseg = False

try:
    from mmdet.models.builder import BACKBONES as det_BACKBONES
    from mmdet.utils import get_root_logger
    from mmcv.runner import _load_checkpoint
    has_mmdet = True
except ImportError:
    has_mmdet = False


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .95, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD, 
        'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    'poolformer_s': _cfg(crop_pct=0.9),
    'poolformer_m': _cfg(crop_pct=0.95),
}


_STAGE_NAMES = ('stage1', 'stage2', 'stage3', 'stage4')

# Experiment-A presets from ``poolformer_s24_reduced_gelu_experiment.md``.
# GELU8 keeps an approximately one-third, spatially distributed subset while
# retaining at least one explicit activation in every stage.
_GELU_PRESETS = {
    'gelu12': {
        'stage1': (True, False, True, False),
        'stage2': (True, False, True, False),
        'stage3': (
            True, False, True, False, True, False,
            True, False, True, False, True, False,
        ),
        'stage4': (True, False, True, False),
    },
    'gelu8': {
        'stage1': (True, False, False, False),
        'stage2': (True, False, False, False),
        'stage3': (
            True, False, False, True, False, False,
            True, False, False, True, False, False,
        ),
        'stage4': (True, False, True, False),
    },
}


def _validate_gelu_mask(gelu_mask, layers):
    expected_keys = set(_STAGE_NAMES)
    actual_keys = set(gelu_mask)
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        extra = sorted(actual_keys - expected_keys)
        raise ValueError(
            'gelu_mask must contain exactly stage1-stage4; '
            f'missing={missing}, extra={extra}')

    validated = {}
    for stage_name, block_count in zip(_STAGE_NAMES, layers):
        stage_mask = tuple(gelu_mask[stage_name])
        if len(stage_mask) != block_count:
            raise ValueError(
                f'{stage_name} GELU mask must have {block_count} entries, '
                f'got {len(stage_mask)}')
        if not all(isinstance(value, bool) for value in stage_mask):
            raise TypeError(f'{stage_name} GELU mask entries must be bools')
        validated[stage_name] = stage_mask
    return validated


def _resolve_gelu_mask(layers, arch_config, gelu_mask):
    arch_config = 'baseline' if arch_config is None else str(arch_config)
    if gelu_mask is not None:
        if arch_config != 'baseline':
            raise ValueError(
                'Specify either a named arch_config or a custom gelu_mask, '
                'not both')
        return 'custom', _validate_gelu_mask(gelu_mask, layers)

    if arch_config == 'baseline':
        return arch_config, {
            stage_name: (True,) * block_count
            for stage_name, block_count in zip(_STAGE_NAMES, layers)
        }
    if arch_config not in _GELU_PRESETS:
        available = ', '.join(('baseline', *_GELU_PRESETS))
        raise ValueError(
            f'Unknown PoolFormer arch_config {arch_config!r}; '
            f'available: {available}')
    return arch_config, _validate_gelu_mask(
        _GELU_PRESETS[arch_config], layers)


def _resolve_block_mlp_ratios(layers, mlp_ratios, block_mlp_ratios):
    """Return a validated ratio tuple for every block in every stage."""
    if len(mlp_ratios) != len(layers):
        raise ValueError(
            f'mlp_ratios must have {len(layers)} stage entries, '
            f'got {len(mlp_ratios)}')
    if block_mlp_ratios is None:
        block_mlp_ratios = {
            stage_name: (stage_ratio,) * block_count
            for stage_name, stage_ratio, block_count in zip(
                _STAGE_NAMES, mlp_ratios, layers)
        }
    else:
        expected_keys = set(_STAGE_NAMES)
        actual_keys = set(block_mlp_ratios)
        if actual_keys != expected_keys:
            missing = sorted(expected_keys - actual_keys)
            extra = sorted(actual_keys - expected_keys)
            raise ValueError(
                'block_mlp_ratios must contain exactly stage1-stage4; '
                f'missing={missing}, extra={extra}')

    validated = {}
    for stage_name, block_count in zip(_STAGE_NAMES, layers):
        stage_ratios = tuple(block_mlp_ratios[stage_name])
        if len(stage_ratios) != block_count:
            raise ValueError(
                f'{stage_name} MLP ratios must have {block_count} entries, '
                f'got {len(stage_ratios)}')
        if not all(
                isinstance(ratio, (int, float)) and ratio > 0
                for ratio in stage_ratios):
            raise ValueError(
                f'{stage_name} MLP ratios must be positive numbers')
        validated[stage_name] = stage_ratios
    return validated


def _format_gelu_mask(gelu_mask):
    return ' '.join(
        f"{stage_name}={''.join('G' if value else '-' for value in gelu_mask[stage_name])}"
        for stage_name in _STAGE_NAMES
    )


class PatchEmbed(nn.Module):
    """
    Patch Embedding that is implemented by a layer of conv. 
    Input: tensor in shape [B, C, H, W]
    Output: tensor in shape [B, C, H/stride, W/stride]
    """
    def __init__(self, patch_size=16, stride=16, padding=0, 
                 in_chans=3, embed_dim=768, norm_layer=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        stride = to_2tuple(stride)
        padding = to_2tuple(padding)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, 
                              stride=stride, padding=padding)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x


class LayerNormChannel(nn.Module):
    """
    LayerNorm only for Channel Dimension.
    Input: tensor in shape [B, C, H, W]
    """
    def __init__(self, num_channels, eps=1e-05):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight.unsqueeze(-1).unsqueeze(-1) * x \
            + self.bias.unsqueeze(-1).unsqueeze(-1)
        return x


class GroupNorm(nn.GroupNorm):
    """
    Group Normalization with 1 group.
    Input: tensor in shape [B, C, H, W]
    """
    def __init__(self, num_channels, **kwargs):
        super().__init__(1, num_channels, **kwargs)


class Pooling(nn.Module):
    """
    Implementation of pooling for PoolFormer
    --pool_size: pooling size
    """
    def __init__(self, pool_size=3):
        super().__init__()
        self.pool = nn.AvgPool2d(
            pool_size, stride=1, padding=pool_size//2, count_include_pad=False)

    def forward(self, x):
        return self.pool(x) - x


class Mlp(nn.Module):
    """
    Implementation of MLP with 1*1 convolutions.
    Input: tensor with shape [B, C, H, W]
    """
    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU, drop=0.,
                 use_activation=True):
        super().__init__()
        if not isinstance(use_activation, bool):
            raise TypeError('use_activation must be a bool')
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer() if use_activation else nn.Identity()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

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
                 act_layer=nn.GELU, norm_layer=GroupNorm, 
                 drop=0., drop_path=0., 
                 use_layer_scale=True, layer_scale_init_value=1e-5,
                 use_activation=True):

        super().__init__()

        self.norm1 = norm_layer(dim)
        self.token_mixer = Pooling(pool_size=pool_size)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
            use_activation=use_activation,
        )

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
                 act_layer=nn.GELU, norm_layer=GroupNorm, 
                 drop_rate=.0, drop_path_rate=0., 
                 use_layer_scale=True, layer_scale_init_value=1e-5,
                 use_activations=None):
    """
    generate PoolFormer blocks for a stage
    return: PoolFormer blocks 
    """
    blocks = []
    block_count = layers[index]
    if isinstance(mlp_ratio, (int, float)):
        mlp_ratio = (mlp_ratio,) * block_count
    else:
        mlp_ratio = tuple(mlp_ratio)
    if len(mlp_ratio) != block_count:
        raise ValueError(
            f'stage{index + 1} MLP ratios must have {block_count} entries, '
            f'got {len(mlp_ratio)}')
    if use_activations is None:
        use_activations = (True,) * block_count
    else:
        use_activations = tuple(use_activations)
    if len(use_activations) != block_count:
        raise ValueError(
            f'stage{index + 1} GELU mask must have {block_count} entries, '
            f'got {len(use_activations)}')
    for block_idx in range(layers[index]):
        block_dpr = drop_path_rate * (
            block_idx + sum(layers[:index])) / (sum(layers) - 1)
        blocks.append(PoolFormerBlock(
            dim, pool_size=pool_size, mlp_ratio=mlp_ratio[block_idx],
            act_layer=act_layer, norm_layer=norm_layer, 
            drop=drop_rate, drop_path=block_dpr, 
            use_layer_scale=use_layer_scale, 
            layer_scale_init_value=layer_scale_init_value,
            use_activation=use_activations[block_idx],
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
                 norm_layer=GroupNorm, act_layer=nn.GELU, 
                 num_classes=512, # modify the num_classes for face recognition
                 in_patch_size=3, in_stride=2, in_pad=1, # modify the patch embedding for face recognition
                 down_patch_size=3, down_stride=2, down_pad=1, 
                 drop_rate=0., drop_path_rate=0.,
                 use_layer_scale=True, layer_scale_init_value=1e-5, 
                 fork_feat=False,
                 face_embedding=True,
                 fp16=False,
                 arch_config='baseline',
                 gelu_mask=None,
                 block_mlp_ratios=None,
                 init_cfg=None, 
                 pretrained=None, 
                 **kwargs):

        super().__init__()

        if not fork_feat:
            self.num_classes = num_classes
        self.fork_feat = fork_feat
        self.face_embedding = face_embedding
        self.fp16 = fp16
        self.arch_config, self.gelu_mask = _resolve_gelu_mask(
            layers, arch_config, gelu_mask)
        self.block_mlp_ratios = _resolve_block_mlp_ratios(
            layers, mlp_ratios, block_mlp_ratios)
        self.gelu_depth = sum(
            sum(self.gelu_mask[stage_name])
            for stage_name in _STAGE_NAMES)
        self.gelu_count_by_stage = {
            stage_name: sum(self.gelu_mask[stage_name])
            for stage_name in _STAGE_NAMES
        }

        self.patch_embed = PatchEmbed(
            patch_size=in_patch_size, stride=in_stride, padding=in_pad, 
            in_chans=3, embed_dim=embed_dims[0])

        # set the main block in network
        network = []
        for i in range(len(layers)):
            stage_name = _STAGE_NAMES[i]
            stage = basic_blocks(embed_dims[i], i, layers,
                                 pool_size=pool_size,
                                 mlp_ratio=self.block_mlp_ratios[stage_name],
                                 act_layer=act_layer, norm_layer=norm_layer, 
                                 drop_rate=drop_rate, 
                                 drop_path_rate=drop_path_rate,
                                 use_layer_scale=use_layer_scale, 
                                 layer_scale_init_value=layer_scale_init_value,
                                 use_activations=self.gelu_mask[stage_name])
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
                    # TODO: more elegant way
                    """For RetinaNet, `start_level=1`. The first norm layer will not used.
                    cmd: `FORK_LAST3=1 python -m torch.distributed.launch ...`
                    """
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
                    nn.Conv2d(embed_dims[-1], embed_dims[-1], kernel_size=(7,7), stride=(1,1), padding=(0,0), groups=1),
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

        if int(os.environ.get('RANK', '0')) == 0:
            print(
                f'PoolFormer arch_config={self.arch_config} '
                f'GELU depth={self.gelu_depth} '
                f'per-stage GELU count={self.gelu_count_by_stage} '
                f'{_format_gelu_mask(self.gelu_mask)}')

    def load_backbone_init_state_dict(self, state_dict):
        """Load an equal-width baseline after GELUs become parameterless identities."""
        translated = dict(state_dict)
        if translated and all(key.startswith('module.') for key in translated):
            translated = {
                key[len('module.'):]: value for key, value in translated.items()
            }
        return self.load_state_dict(translated, strict=True)

    # init for classification
    def cls_init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    # init for mmdetection or mmsegmentation by loading 
    # imagenet pre-trained weights
    def init_weights(self, pretrained=None):
        logger = get_root_logger()
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

            ckpt = _load_checkpoint(
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
            
            # show for debug
            # print('missing_keys: ', missing_keys)
            # print('unexpected_keys: ', unexpected_keys)

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
            # otuput features of four stages for dense prediction
            return x
        x = self.norm(x)
        if self.face_embedding:
            return self.head(x.float() if self.fp16 else x)
        x = x.mean([-2, -1])
        cls_out = self.head(x)
        # for image classification
        return cls_out


model_urls = {
    "poolformer_s12": "https://github.com/sail-sg/poolformer/releases/download/v1.0/poolformer_s12.pth.tar",
    "poolformer_s24": "https://github.com/sail-sg/poolformer/releases/download/v1.0/poolformer_s24.pth.tar",
    "poolformer_s36": "https://github.com/sail-sg/poolformer/releases/download/v1.0/poolformer_s36.pth.tar",
    "poolformer_m36": "https://github.com/sail-sg/poolformer/releases/download/v1.0/poolformer_m36.pth.tar",
    "poolformer_m48": "https://github.com/sail-sg/poolformer/releases/download/v1.0/poolformer_m48.pth.tar",
}


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
    if pretrained:
        url = model_urls['poolformer_s12']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint)
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
    if pretrained:
        url = model_urls['poolformer_s24']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint)
    return model


def _poolformer_s24_reduced_gelu(preset, pretrained=False, **kwargs):
    if pretrained:
        raise ValueError(
            'Reduced-GELU face backbones do not provide pretrained weights')
    configured_arch = kwargs.pop('arch_config', preset)
    if configured_arch != preset:
        raise ValueError(
            f'poolformer_s24_{preset} requires '
            f'arch_config={preset!r}, got {configured_arch!r}')
    return poolformer_s24(
        pretrained=False,
        arch_config=preset,
        **kwargs,
    )


@register_model
def poolformer_s24_gelu12(pretrained=False, **kwargs):
    """PoolFormer-S24 retaining the mandatory alternating 12-GELU mask."""
    return _poolformer_s24_reduced_gelu(
        'gelu12', pretrained=pretrained, **kwargs)


@register_model
def poolformer_s24_gelu8(pretrained=False, **kwargs):
    """PoolFormer-S24 retaining eight GELUs distributed across all stages."""
    return _poolformer_s24_reduced_gelu(
        'gelu8', pretrained=pretrained, **kwargs)


@register_model
def poolformer_s24_mlp2(pretrained=False, **kwargs):
    """
    PoolFormer-S24 model with MLP ratios [2, 2, 2, 2].
    """
    layers = [4, 4, 12, 4]
    embed_dims = [64, 128, 320, 512]
    mlp_ratios = [2, 2, 2, 2]
    downsamples = [True, True, True, True]
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
    if pretrained:
        url = model_urls['poolformer_s36']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint)
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
    if pretrained:
        url = model_urls['poolformer_m36']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint)
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
    if pretrained:
        url = model_urls['poolformer_m48']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint)
    return model


if has_mmseg and has_mmdet:
    """
    The following models are for dense prediction based on 
    mmdetection and mmsegmentation
    """
    @seg_BACKBONES.register_module()
    @det_BACKBONES.register_module()
    class poolformer_s12_feat(PoolFormer):
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

    @seg_BACKBONES.register_module()
    @det_BACKBONES.register_module()
    class poolformer_s24_feat(PoolFormer):
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

    @seg_BACKBONES.register_module()
    @det_BACKBONES.register_module()
    class poolformer_s36_feat(PoolFormer):
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

    @seg_BACKBONES.register_module()
    @det_BACKBONES.register_module()
    class poolformer_m36_feat(PoolFormer):
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

    @seg_BACKBONES.register_module()
    @det_BACKBONES.register_module()
    class poolformer_m48_feat(PoolFormer):
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

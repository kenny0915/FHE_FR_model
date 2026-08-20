import os

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

__all__ = ['iresnet18', 'iresnet34', 'iresnet50', 'iresnet100', 'iresnet200']
using_ckpt = False

_STAGE_NAMES = ('stage1', 'stage2', 'stage3', 'stage4')
_STAGE_CHANNELS = (64, 128, 256, 512)

# The first reduced-nonlinearity experiment retains every stage-transition
# activation and distributes the remaining activations through the long stage.
_ACTIVATION_PRESETS = {
    'nl13': {
        'stem': True,
        'stage1': (True, False, True),
        'stage2': (True, False, False, True),
        'stage3': (
            True, False, False, True, False, False, True,
            False, False, True, False, False, False, True,
        ),
        'stage4': (True, True, True),
    },
}


def _validate_activation_mask(activation_mask, layers):
    expected_keys = {'stem', *_STAGE_NAMES}
    actual_keys = set(activation_mask)
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        extra = sorted(actual_keys - expected_keys)
        raise ValueError(
            'activation_mask must contain exactly stem and stage1-stage4; '
            f'missing={missing}, extra={extra}')
    if not isinstance(activation_mask['stem'], bool):
        raise TypeError('activation_mask["stem"] must be a bool')

    validated = {'stem': activation_mask['stem']}
    for stage_name, block_count in zip(_STAGE_NAMES, layers):
        stage_mask = tuple(activation_mask[stage_name])
        if len(stage_mask) != block_count:
            raise ValueError(
                f'{stage_name} activation mask must have {block_count} entries, '
                f'got {len(stage_mask)}')
        if not all(isinstance(value, bool) for value in stage_mask):
            raise TypeError(f'{stage_name} activation mask entries must be bools')
        validated[stage_name] = stage_mask
    return validated


def _resolve_activation_mask(layers, arch_config, activation_mask):
    arch_config = 'baseline' if arch_config is None else str(arch_config)
    if activation_mask is not None:
        if arch_config != 'baseline':
            raise ValueError(
                'Specify either a named arch_config or a custom activation_mask, '
                'not both')
        return 'custom', _validate_activation_mask(activation_mask, layers)

    if arch_config == 'baseline':
        mask = {'stem': True}
        mask.update({
            stage_name: (True,) * block_count
            for stage_name, block_count in zip(_STAGE_NAMES, layers)
        })
        return arch_config, mask
    if arch_config not in _ACTIVATION_PRESETS:
        available = ', '.join(('baseline', *_ACTIVATION_PRESETS))
        raise ValueError(
            f'Unknown IResNet arch_config {arch_config!r}; available: {available}')

    return arch_config, _validate_activation_mask(
        _ACTIVATION_PRESETS[arch_config], layers)


def _resolve_mid_widths(layers, mid_widths):
    if mid_widths is None:
        return {
            stage_name: (channels,) * block_count
            for stage_name, channels, block_count in zip(
                _STAGE_NAMES, _STAGE_CHANNELS, layers)
        }

    expected_keys = set(_STAGE_NAMES)
    actual_keys = set(mid_widths)
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        extra = sorted(actual_keys - expected_keys)
        raise ValueError(
            'mid_widths must contain exactly stage1-stage4; '
            f'missing={missing}, extra={extra}')

    validated = {}
    for stage_name, block_count in zip(_STAGE_NAMES, layers):
        stage_widths = tuple(mid_widths[stage_name])
        if len(stage_widths) != block_count:
            raise ValueError(
                f'{stage_name} mid_widths must have {block_count} entries, '
                f'got {len(stage_widths)}')
        if not all(isinstance(width, int) and width > 0
                   for width in stage_widths):
            raise ValueError(f'{stage_name} mid_widths must be positive integers')
        validated[stage_name] = stage_widths
    return validated


def _format_activation_mask(activation_mask):
    def symbols(values):
        return ''.join('P' if value else '-' for value in values)

    return ' '.join([
        f"stem={'P' if activation_mask['stem'] else '-'}",
        *(f'{stage_name}={symbols(activation_mask[stage_name])}'
          for stage_name in _STAGE_NAMES),
    ])

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=dilation,
                     groups=groups,
                     bias=False,
                     dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


class IBasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 groups=1, base_width=64, dilation=1, mid_channels=None,
                 use_activation=True):
        super(IBasicBlock, self).__init__()
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        if mid_channels is None:
            mid_channels = planes
        if not isinstance(mid_channels, int) or mid_channels <= 0:
            raise ValueError('mid_channels must be a positive integer')
        if not isinstance(use_activation, bool):
            raise TypeError('use_activation must be a bool')
        self.mid_channels = mid_channels
        self.use_activation = use_activation
        self.bn1 = nn.BatchNorm2d(inplanes, eps=1e-05,)
        self.conv1 = conv3x3(inplanes, mid_channels)
        self.bn2 = nn.BatchNorm2d(mid_channels, eps=1e-05,)
        self.prelu = (
            nn.PReLU(mid_channels) if use_activation else nn.Identity())
        self.conv2 = conv3x3(mid_channels, planes, stride)
        self.bn3 = nn.BatchNorm2d(planes, eps=1e-05,)
        self.downsample = downsample
        self.stride = stride

    def forward_impl(self, x):
        identity = x
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return out        

    def forward(self, x):
        if self.training and using_ckpt:
            return checkpoint(self.forward_impl, x)
        else:
            return self.forward_impl(x)


class IResNet(nn.Module):
    fc_scale = 7 * 7
    def __init__(self,
                 block, layers, dropout=0, num_features=512, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 fp16=False, arch_config='baseline', activation_mask=None,
                 mid_widths=None):
        super(IResNet, self).__init__()
        self.extra_gflops = 0.0
        self.fp16 = fp16
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.arch_config, self.activation_mask = _resolve_activation_mask(
            layers, arch_config, activation_mask)
        self.mid_widths = _resolve_mid_widths(layers, mid_widths)
        self.nonlinear_depth = int(self.activation_mask['stem']) + sum(
            sum(stage_mask)
            for stage_mask in (
                self.activation_mask[stage_name]
                for stage_name in _STAGE_NAMES)
        )
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes, eps=1e-05)
        self.prelu = (
            nn.PReLU(self.inplanes)
            if self.activation_mask['stem'] else nn.Identity())
        self.layer1 = self._make_layer(
            block, 64, layers[0], stride=2,
            activation_mask=self.activation_mask['stage1'],
            mid_widths=self.mid_widths['stage1'])
        self.layer2 = self._make_layer(block,
                                       128,
                                       layers[1],
                                       stride=2,
                                       dilate=replace_stride_with_dilation[0],
                                       activation_mask=self.activation_mask['stage2'],
                                       mid_widths=self.mid_widths['stage2'])
        self.layer3 = self._make_layer(block,
                                       256,
                                       layers[2],
                                       stride=2,
                                       dilate=replace_stride_with_dilation[1],
                                       activation_mask=self.activation_mask['stage3'],
                                       mid_widths=self.mid_widths['stage3'])
        self.layer4 = self._make_layer(block,
                                       512,
                                       layers[3],
                                       stride=2,
                                       dilate=replace_stride_with_dilation[2],
                                       activation_mask=self.activation_mask['stage4'],
                                       mid_widths=self.mid_widths['stage4'])
        self.bn2 = nn.BatchNorm2d(512 * block.expansion, eps=1e-05,)
        self.dropout = nn.Dropout(p=dropout, inplace=True)
        self.fc = nn.Linear(512 * block.expansion * self.fc_scale, num_features)
        self.features = nn.BatchNorm1d(num_features, eps=1e-05)
        nn.init.constant_(self.features.weight, 1.0)
        self.features.weight.requires_grad = False

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.1)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, IBasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

        if int(os.environ.get('RANK', '0')) == 0:
            print(
                f'IResNet arch_config={self.arch_config} '
                f'nonlinear_depth={self.nonlinear_depth} '
                f'{_format_activation_mask(self.activation_mask)}')

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False,
                    activation_mask=None, mid_widths=None):
        if activation_mask is None:
            activation_mask = (True,) * blocks
        if mid_widths is None:
            mid_widths = (planes,) * blocks
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion, eps=1e-05, ),
            )
        layers = []
        layers.append(
            block(self.inplanes, planes, stride, downsample, self.groups,
                  self.base_width, previous_dilation,
                  mid_channels=mid_widths[0],
                  use_activation=activation_mask[0]))
        self.inplanes = planes * block.expansion
        for block_index in range(1, blocks):
            layers.append(
                block(self.inplanes,
                      planes,
                      groups=self.groups,
                      base_width=self.base_width,
                      dilation=self.dilation,
                      mid_channels=mid_widths[block_index],
                      use_activation=activation_mask[block_index]))

        return nn.Sequential(*layers)

    def load_backbone_init_state_dict(self, state_dict):
        """Load a baseline checkpoint while ignoring removed PReLU weights."""
        translated = dict(state_dict)
        if translated and all(key.startswith('module.') for key in translated):
            translated = {
                key[len('module.'):]: value for key, value in translated.items()
            }

        for module_name, module in self.named_modules():
            if isinstance(module, nn.Identity) and (
                    module_name == 'prelu' or module_name.endswith('.prelu')):
                translated.pop(f'{module_name}.weight', None)

        return self.load_state_dict(translated, strict=True)

    def forward(self, x):
        with torch.cuda.amp.autocast(self.fp16):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.prelu(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.bn2(x)
            x = torch.flatten(x, 1)
            x = self.dropout(x)
        x = self.fc(x.float() if self.fp16 else x)
        x = self.features(x)
        return x


def _iresnet(arch, block, layers, pretrained, progress, **kwargs):
    model = IResNet(block, layers, **kwargs)
    if pretrained:
        raise ValueError()
    return model


def iresnet18(pretrained=False, progress=True, **kwargs):
    return _iresnet('iresnet18', IBasicBlock, [2, 2, 2, 2], pretrained,
                    progress, **kwargs)


def iresnet34(pretrained=False, progress=True, **kwargs):
    return _iresnet('iresnet34', IBasicBlock, [3, 4, 6, 3], pretrained,
                    progress, **kwargs)


def iresnet50(pretrained=False, progress=True, **kwargs):
    return _iresnet('iresnet50', IBasicBlock, [3, 4, 14, 3], pretrained,
                    progress, **kwargs)


def iresnet100(pretrained=False, progress=True, **kwargs):
    return _iresnet('iresnet100', IBasicBlock, [3, 13, 30, 3], pretrained,
                    progress, **kwargs)


def iresnet200(pretrained=False, progress=True, **kwargs):
    return _iresnet('iresnet200', IBasicBlock, [6, 26, 60, 6], pretrained,
                    progress, **kwargs)

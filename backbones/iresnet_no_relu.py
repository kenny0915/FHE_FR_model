import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

__all__ = [
    'iresnet18', 'iresnet34', 'iresnet50', 'iresnet100', 'iresnet200',
    'ChebyReLU', 'ProgressiveChebyActivation',
]
using_ckpt = False

_CHEBY_STAGE_NAMES = ('stem', 'layer1', 'layer2', 'layer3', 'layer4')
_DEFAULT_CHEBY_SCALES = {
    'stem': 8.0,
    'layer1': 8.0,
    'layer2': 7.0,
    'layer3': 6.5,
    'layer4': 6.5,
}


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


class ChebyReLU(nn.Module):
    _NORMALIZED_POWER_COEFFS = (1.05146222424, -0.581234022404)

    def __init__(self, input_scale=8.0):
        super().__init__()
        if input_scale <= 0:
            raise ValueError('input_scale must be positive')
        self.register_buffer(
            'input_scale', torch.tensor(float(input_scale), dtype=torch.float32))
        self.register_buffer(
            'normalized_power_coeffs',
            torch.tensor(self._NORMALIZED_POWER_COEFFS, dtype=torch.float32))

    def forward(self, x):
        compute_dtype = (
            torch.float32
            if x.dtype in (torch.float16, torch.bfloat16)
            else x.dtype
        )
        compute_x = x.to(dtype=compute_dtype)
        scale = self.input_scale.to(device=x.device, dtype=compute_dtype)
        coefficients = self.normalized_power_coeffs.to(
            device=x.device, dtype=compute_dtype)

        z = compute_x / scale
        z_squared = z * z
        z_fourth = z_squared * z_squared
        even_part = coefficients[0] * z_squared + coefficients[1] * z_fourth
        out = 0.5 * compute_x + scale * even_part
        return out.to(dtype=x.dtype)


class ProgressiveChebyActivation(nn.Module):
    """Training-only PReLU-to-Cheby transition with range regularization.

    At blend=0 this is a trainable PReLU, and at blend=1 its eval-time
    forward is exactly ChebyReLU. Intermediate blends make it possible to
    convert one IResNet stage at a time without abruptly changing every
    activation distribution.
    """

    def __init__(self, channels, input_scale, range_limit=6.0,
                 stage_index=0, blend=1.0):
        super().__init__()
        if range_limit <= 0:
            raise ValueError('range_limit must be positive')
        self.prelu = nn.PReLU(channels)
        self.cheby = ChebyReLU(input_scale=input_scale)
        self.stage_index = int(stage_index)
        self.register_buffer('blend', torch.tensor(float(blend), dtype=torch.float32))
        self.register_buffer(
            'range_limit', torch.tensor(float(range_limit), dtype=torch.float32))
        self._last_range_penalty = None
        self._last_input_absmax = None
        self._last_outside_fraction = None
        self._blend = 0.0
        self.set_blend(blend)

    def set_blend(self, blend):
        blend = float(blend)
        if not 0.0 <= blend <= 1.0:
            raise ValueError('blend must be in [0, 1]')
        self._blend = blend
        self.blend.fill_(blend)

    def range_penalty(self):
        return self._last_range_penalty

    def range_stats(self):
        return {
            'absmax': self._last_input_absmax,
            'outside_fraction': self._last_outside_fraction,
            'scale': self.cheby.input_scale.detach(),
            'blend': self.blend.detach(),
        }

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # Accept both the old direct-Cheby checkpoints (input_scale/coeffs)
        # and ordinary IResNet PReLU checkpoints (weight).
        legacy_to_new = {
            prefix + 'weight': prefix + 'prelu.weight',
            prefix + 'input_scale': prefix + 'cheby.input_scale',
            prefix + 'normalized_power_coeffs': prefix + 'cheby.normalized_power_coeffs',
        }
        for old_key, new_key in legacy_to_new.items():
            if old_key in state_dict and new_key not in state_dict:
                state_dict[new_key] = state_dict.pop(old_key)

        defaults = {
            prefix + 'prelu.weight': self.prelu.weight.detach(),
            prefix + 'cheby.input_scale': self.cheby.input_scale.detach(),
            prefix + 'cheby.normalized_power_coeffs':
                self.cheby.normalized_power_coeffs.detach(),
            prefix + 'blend': self.blend.detach(),
            prefix + 'range_limit': self.range_limit.detach(),
        }
        for key, value in defaults.items():
            if key not in state_dict:
                state_dict[key] = value
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)
        self._blend = float(self.blend.item())

    def forward(self, x):
        if self.training:
            compute_x = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
            limit = self.range_limit.to(device=x.device, dtype=compute_x.dtype)
            excess = torch.relu(compute_x.abs() - limit)
            # This loss is consumed by train_v2.py and is absent from the
            # final inference graph.
            self._last_range_penalty = (excess * excess).mean()
            self._last_input_absmax = compute_x.detach().abs().amax()
            self._last_outside_fraction = (excess.detach() > 0).float().mean()
        else:
            self._last_range_penalty = None

        blend = self._blend
        if blend <= 0.0:
            return self.prelu(x)

        cheby_out = self.cheby(x)
        if blend >= 1.0:
            if self.training:
                # Keep the PReLU parameter in DDP's graph while its
                # contribution is zero. In eval mode the path is polynomial-only.
                return cheby_out + self.prelu(x) * 0.0
            return cheby_out
        return (1.0 - blend) * self.prelu(x) + blend * cheby_out


def _normalize_cheby_scales(scales):
    if scales is None:
        return dict(_DEFAULT_CHEBY_SCALES)
    if isinstance(scales, (int, float)):
        values = {name: float(scales) for name in _CHEBY_STAGE_NAMES}
    elif isinstance(scales, dict):
        values = dict(_DEFAULT_CHEBY_SCALES)
        values.update({name: float(value) for name, value in scales.items()})
    else:
        if len(scales) != len(_CHEBY_STAGE_NAMES):
            raise ValueError('cheby_scales must have {} values'.format(
                len(_CHEBY_STAGE_NAMES)))
        values = dict(zip(_CHEBY_STAGE_NAMES, map(float, scales)))
    if any(value <= 0 for value in values.values()):
        raise ValueError('all Cheby input scales must be positive')
    return values


class IBasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 groups=1, base_width=64, dilation=1,
                 activation_factory=None):
        super(IBasicBlock, self).__init__()
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.bn1 = nn.BatchNorm2d(inplanes, eps=1e-05,)
        self.conv1 = conv3x3(inplanes, planes)
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-05,)
        if activation_factory is None:
            activation_factory = lambda channels: ChebyReLU(input_scale=8.0)
        self.prelu = activation_factory(planes)
        self.conv2 = conv3x3(planes, planes, stride)
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
                 fp16=False, cheby_scales=None, cheby_range_limit=6.0,
                 cheby_progress=5.0):
        super(IResNet, self).__init__()
        self.extra_gflops = 0.0
        self.fp16 = fp16
        self.inplanes = 64
        self.dilation = 1
        self.cheby_scales = _normalize_cheby_scales(cheby_scales)
        self.cheby_range_limit = float(cheby_range_limit)
        self.register_buffer(
            'cheby_progress', torch.tensor(float(cheby_progress), dtype=torch.float32),
            persistent=False)
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes, eps=1e-05)
        self.prelu = self._make_activation(self.inplanes, 'stem')
        self.layer1 = self._make_layer(block, 64, layers[0], stride=2,
                                       stage_name='layer1')
        self.layer2 = self._make_layer(block,
                                       128,
                                       layers[1],
                                       stride=2,
                                       dilate=replace_stride_with_dilation[0],
                                       stage_name='layer2')
        self.layer3 = self._make_layer(block,
                                       256,
                                       layers[2],
                                       stride=2,
                                       dilate=replace_stride_with_dilation[1],
                                       stage_name='layer3')
        self.layer4 = self._make_layer(block,
                                       512,
                                       layers[3],
                                       stride=2,
                                       dilate=replace_stride_with_dilation[2],
                                       stage_name='layer4')
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

        self.set_cheby_progress(cheby_progress)

    def _make_activation(self, channels, stage_name, activation_name=None):
        stage_index = _CHEBY_STAGE_NAMES.index(stage_name)
        input_scale = self.cheby_scales.get(
            activation_name, self.cheby_scales[stage_name])
        return ProgressiveChebyActivation(
            channels=channels,
            input_scale=input_scale,
            range_limit=self.cheby_range_limit,
            stage_index=stage_index,
            blend=0.0,
        )

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False,
                    stage_name=None):
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
        activation_factory = lambda channels: self._make_activation(
            channels, stage_name, '{}.0.prelu'.format(stage_name))
        layers.append(
            block(self.inplanes, planes, stride, downsample, self.groups,
                  self.base_width, previous_dilation,
                  activation_factory=activation_factory))
        self.inplanes = planes * block.expansion
        for block_index in range(1, blocks):
            activation_factory = lambda channels, index=block_index: self._make_activation(
                channels, stage_name,
                '{}.{}.prelu'.format(stage_name, index))
            layers.append(
                block(self.inplanes,
                      planes,
                      groups=self.groups,
                      base_width=self.base_width,
                      dilation=self.dilation,
                      activation_factory=activation_factory))

        return nn.Sequential(*layers)

    def progressive_activations(self):
        return [module for module in self.modules()
                if isinstance(module, ProgressiveChebyActivation)]

    def set_cheby_progress(self, progress):
        """Set conversion progress: 0=PReLU, 5=all five stages Cheby."""
        progress = min(max(float(progress), 0.0), float(len(_CHEBY_STAGE_NAMES)))
        self.cheby_progress.fill_(progress)
        for activation in self.progressive_activations():
            activation.set_blend(min(max(progress - activation.stage_index, 0.0), 1.0))

    def cheby_range_penalty(self):
        penalties = [activation.range_penalty()
                     for activation in self.progressive_activations()]
        penalties = [penalty for penalty in penalties if penalty is not None]
        if not penalties:
            return next(self.parameters()).new_zeros(())
        return torch.stack(penalties).mean()

    def cheby_range_stats(self):
        stats = {}
        for name, module in self.named_modules():
            if isinstance(module, ProgressiveChebyActivation):
                stats[name] = module.range_stats()
        return stats

    def cheby_range_summary(self):
        stats = list(self.cheby_range_stats().values())
        absmax = [item['absmax'] for item in stats if item['absmax'] is not None]
        outside = [item['outside_fraction'] for item in stats
                   if item['outside_fraction'] is not None]
        zero = next(self.parameters()).new_zeros(())
        return {
            'input_absmax': torch.stack(absmax).amax() if absmax else zero,
            'outside_fraction': torch.stack(outside).mean() if outside else zero,
        }

    def begin_batchnorm_recalibration(self, reset=True):
        """Put only BatchNorm layers in train mode and optionally reset stats."""
        batchnorm_state = [
            (module, module.training, module.momentum)
            for module in self.modules()
            if isinstance(module, nn.modules.batchnorm._BatchNorm)
        ]
        state = {
            'model_training': self.training,
            'batchnorm': batchnorm_state,
        }
        self.eval()
        for module, _, _ in batchnorm_state:
            if reset:
                module.reset_running_stats()
            module.momentum = None
            module.train()
        return state

    def end_batchnorm_recalibration(self, state):
        self.train(state['model_training'])
        for module, was_training, momentum in state['batchnorm']:
            module.momentum = momentum
            module.train(was_training)

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

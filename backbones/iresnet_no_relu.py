import math

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

__all__ = [
    'iresnet18', 'iresnet34', 'iresnet50', 'iresnet100', 'iresnet200',
    'HerPN', 'FoldedHerPN', 'ProgressiveHerPNActivation',
]
using_ckpt = False

_HERPN_STAGE_NAMES = ('stem', 'layer1', 'layer2', 'layer3', 'layer4')


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding."""
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=dilation,
                     groups=groups,
                     bias=False,
                     dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution."""
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


class HerPN(nn.Module):
    """CryptoFace's normalized degree-2 Hermite polynomial activation.

    The three non-affine BatchNorms normalize the Hermite bases during
    training. Once their running statistics are calibrated, this module is
    exactly a channel-wise quadratic A*x^2 + B*x + C at inference time.
    """

    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.bn0 = nn.BatchNorm2d(channels, eps=eps, affine=False)
        self.bn1 = nn.BatchNorm2d(channels, eps=eps, affine=False)
        self.bn2 = nn.BatchNorm2d(channels, eps=eps, affine=False)
        self.weight = nn.Parameter(torch.ones(channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(channels, 1, 1))

    def forward(self, x):
        compute_dtype = (
            torch.float32
            if x.dtype in (torch.float16, torch.bfloat16)
            else x.dtype
        )
        compute_x = x.to(dtype=compute_dtype)
        x0 = self.bn0(torch.ones_like(compute_x))
        x1 = self.bn1(compute_x)
        x2 = self.bn2((compute_x.square() - 1.0) / math.sqrt(2.0))
        basis = (
            x0 / math.sqrt(2.0 * math.pi)
            + x1 / 2.0
            + x2 / math.sqrt(4.0 * math.pi)
        )
        out = self.weight.to(dtype=compute_dtype) * basis
        out = out + self.bias.to(dtype=compute_dtype)
        return out.to(dtype=x.dtype)

    @torch.no_grad()
    def folded_coefficients(self):
        """Return coefficients for the exactly equivalent eval polynomial."""
        if self.training or self.bn0.training or self.bn1.training or self.bn2.training:
            raise RuntimeError('HerPN must be in eval mode before it can be folded')

        mean0, var0 = self.bn0.running_mean, self.bn0.running_var
        mean1, var1 = self.bn1.running_mean, self.bn1.running_var
        mean2, var2 = self.bn2.running_mean, self.bn2.running_var
        weight = self.weight.squeeze(-1).squeeze(-1)
        bias = self.bias.squeeze(-1).squeeze(-1)
        eps0, eps1, eps2 = self.bn0.eps, self.bn1.eps, self.bn2.eps

        coefficient2 = weight / torch.sqrt(8.0 * math.pi * (var2 + eps2))
        coefficient1 = weight / (2.0 * torch.sqrt(var1 + eps1))
        coefficient0 = bias + weight * (
            (1.0 - mean0) / torch.sqrt(2.0 * math.pi * (var0 + eps0))
            - mean1 / (2.0 * torch.sqrt(var1 + eps1))
            - (1.0 + math.sqrt(2.0) * mean2)
            / torch.sqrt(8.0 * math.pi * (var2 + eps2))
        )
        return tuple(
            coefficient.unsqueeze(-1).unsqueeze(-1)
            for coefficient in (coefficient2, coefficient1, coefficient0)
        )


class FoldedHerPN(nn.Module):
    """Inference-only quadratic produced by folding a calibrated HerPN."""

    def __init__(self, coefficient2, coefficient1, coefficient0):
        super().__init__()
        self.register_buffer('coefficient2', coefficient2.detach().clone())
        self.register_buffer('coefficient1', coefficient1.detach().clone())
        self.register_buffer('coefficient0', coefficient0.detach().clone())

    @classmethod
    def from_herpn(cls, herpn):
        return cls(*herpn.folded_coefficients())

    def forward(self, x):
        compute_dtype = (
            torch.float32
            if x.dtype in (torch.float16, torch.bfloat16)
            else x.dtype
        )
        compute_x = x.to(dtype=compute_dtype)
        coefficient2 = self.coefficient2.to(dtype=compute_dtype)
        coefficient1 = self.coefficient1.to(dtype=compute_dtype)
        coefficient0 = self.coefficient0.to(dtype=compute_dtype)
        out = coefficient2 * compute_x.square() + coefficient1 * compute_x
        return (out + coefficient0).to(dtype=x.dtype)


class ProgressiveHerPNActivation(nn.Module):
    """PReLU-compatible wrapper for a staged PReLU-to-HerPN conversion."""

    def __init__(self, channels, range_limit=6.0, bn_eps=1e-4,
                 stage_index=0, blend=1.0):
        super().__init__()
        if range_limit <= 0:
            raise ValueError('range_limit must be positive')
        self.prelu = nn.PReLU(channels)
        self.herpn = HerPN(channels, eps=bn_eps)
        self.stage_index = int(stage_index)
        self.register_buffer('blend', torch.tensor(float(blend), dtype=torch.float32))
        self.register_buffer(
            'range_limit', torch.tensor(float(range_limit), dtype=torch.float32))
        self._last_range_penalty = None
        self._last_distillation_loss = None
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

    def distillation_loss(self):
        return self._last_distillation_loss

    def range_stats(self):
        return {
            'absmax': self._last_input_absmax,
            'outside_fraction': self._last_outside_fraction,
            'blend': self.blend.detach(),
        }

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # An ordinary IResNet checkpoint stores e.g. layer1.0.prelu.weight.
        # Preserve strict loading by moving it into this wrapper's PReLU.
        old_key = prefix + 'weight'
        new_key = prefix + 'prelu.weight'
        if old_key in state_dict and new_key not in state_dict:
            state_dict[new_key] = state_dict.pop(old_key)

        # A PReLU checkpoint naturally has no HerPN state. Initialize all new
        # state only in that case. If any HerPN key exists, strict=True still
        # reports a partially written/corrupt HerPN checkpoint.
        has_herpn_state = any(
            key.startswith(prefix + 'herpn.') for key in state_dict)
        if not has_herpn_state:
            for local_key, value in self.state_dict().items():
                full_key = prefix + local_key
                if full_key not in state_dict:
                    state_dict[full_key] = value.detach()

        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)
        self._blend = float(self.blend.item())

    def forward(self, x):
        if self.training:
            compute_x = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
            limit = self.range_limit.to(device=x.device, dtype=compute_x.dtype)
            excess = torch.relu(compute_x.abs() - limit)
            mean_tail = excess.square().mean()
            sample_tail = excess.flatten(1).amax(dim=1).square().mean()
            self._last_range_penalty = mean_tail + 0.1 * sample_tail
            self._last_input_absmax = compute_x.detach().abs().amax()
            self._last_outside_fraction = (excess.detach() > 0).float().mean()
        else:
            self._last_range_penalty = None
            self._last_distillation_loss = None

        blend = self._blend
        if not self.training and blend <= 0.0:
            return self.prelu(x)

        # Evaluate both branches in training even at the endpoints. This warms
        # every HerPN BN before conversion and keeps DDP's graph stationary.
        prelu_out = self.prelu(x)
        herpn_out = self.herpn(x)
        if self.training:
            target = prelu_out.detach().float()
            self._last_distillation_loss = (
                (1.0 - blend) * (herpn_out.float() - target).square().mean()
            )

        if blend <= 0.0:
            return prelu_out + herpn_out * 0.0
        if blend >= 1.0:
            return herpn_out + prelu_out * 0.0 if self.training else herpn_out
        return (1.0 - blend) * prelu_out + blend * herpn_out

    def folded(self):
        if self._blend < 1.0:
            raise RuntimeError('Only a fully converted HerPN activation can be folded')
        return FoldedHerPN.from_herpn(self.herpn)


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
        self.bn1 = nn.BatchNorm2d(inplanes, eps=1e-5)
        self.conv1 = conv3x3(inplanes, planes)
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-5)
        if activation_factory is None:
            activation_factory = HerPN
        # Keep the original attribute name so the IResNet topology and old
        # PReLU checkpoint locations do not change.
        self.prelu = activation_factory(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn3 = nn.BatchNorm2d(planes, eps=1e-5)
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
        return self.forward_impl(x)


class IResNet(nn.Module):
    fc_scale = 7 * 7

    def __init__(self,
                 block, layers, dropout=0, num_features=512, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 fp16=False, herpn_range_limit=6.0, herpn_bn_eps=1e-4,
                 herpn_progress=5.0):
        super(IResNet, self).__init__()
        self.extra_gflops = 0.0
        self.fp16 = fp16
        self.inplanes = 64
        self.dilation = 1
        self.herpn_range_limit = float(herpn_range_limit)
        self.herpn_bn_eps = float(herpn_bn_eps)
        self.register_buffer(
            'herpn_progress', torch.tensor(float(herpn_progress), dtype=torch.float32),
            persistent=False)
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(
                                 replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(
            3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes, eps=1e-5)
        self.prelu = self._make_activation(self.inplanes, 'stem')
        self.layer1 = self._make_layer(
            block, 64, layers[0], stride=2, stage_name='layer1')
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2,
            dilate=replace_stride_with_dilation[0], stage_name='layer2')
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=2,
            dilate=replace_stride_with_dilation[1], stage_name='layer3')
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=2,
            dilate=replace_stride_with_dilation[2], stage_name='layer4')
        self.bn2 = nn.BatchNorm2d(512 * block.expansion, eps=1e-5)
        self.dropout = nn.Dropout(p=dropout, inplace=True)
        self.fc = nn.Linear(512 * block.expansion * self.fc_scale, num_features)
        self.features = nn.BatchNorm1d(num_features, eps=1e-5)
        nn.init.constant_(self.features.weight, 1.0)
        self.features.weight.requires_grad = False

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, 0, 0.1)
            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                if module.weight is not None:
                    nn.init.constant_(module.weight, 1)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        if zero_init_residual:
            for module in self.modules():
                if isinstance(module, IBasicBlock):
                    nn.init.constant_(module.bn2.weight, 0)

        self.set_herpn_progress(herpn_progress)

    def _make_activation(self, channels, stage_name):
        return ProgressiveHerPNActivation(
            channels=channels,
            range_limit=self.herpn_range_limit,
            bn_eps=self.herpn_bn_eps,
            stage_index=_HERPN_STAGE_NAMES.index(stage_name),
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
                nn.BatchNorm2d(planes * block.expansion, eps=1e-5),
            )
        layers = []
        activation_factory = lambda channels: self._make_activation(
            channels, stage_name)
        layers.append(
            block(self.inplanes, planes, stride, downsample, self.groups,
                  self.base_width, previous_dilation,
                  activation_factory=activation_factory))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            activation_factory = lambda channels: self._make_activation(
                channels, stage_name)
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
                if isinstance(module, ProgressiveHerPNActivation)]

    def set_herpn_progress(self, progress):
        """Set conversion progress: 0=PReLU, 5=all five stages HerPN."""
        progress = min(max(float(progress), 0.0), float(len(_HERPN_STAGE_NAMES)))
        self.herpn_progress.fill_(progress)
        for activation in self.progressive_activations():
            activation.set_blend(
                min(max(progress - activation.stage_index, 0.0), 1.0))

    def set_herpn_blends(self, blends):
        """Set per-activation blends for schedules that split a large stage."""
        activations = {
            name: module for name, module in self.named_modules()
            if isinstance(module, ProgressiveHerPNActivation)
        }
        unknown = sorted(set(blends).difference(activations))
        if unknown:
            raise ValueError('Unknown HerPN activation names: {}'.format(unknown))
        for name, activation in activations.items():
            activation.set_blend(float(blends.get(name, 0.0)))
        converted_fraction = sum(
            activation._blend for activation in activations.values()
        ) / len(activations)
        self.herpn_progress.fill_(converted_fraction * len(_HERPN_STAGE_NAMES))

    def herpn_range_penalty(self):
        penalties = [activation.range_penalty()
                     for activation in self.progressive_activations()]
        penalties = [penalty for penalty in penalties if penalty is not None]
        if not penalties:
            return next(self.parameters()).new_zeros(())
        return torch.stack(penalties).mean()

    def herpn_distillation_loss(self):
        losses = [activation.distillation_loss()
                  for activation in self.progressive_activations()]
        losses = [loss for loss in losses if loss is not None]
        if not losses:
            return next(self.parameters()).new_zeros(())
        return torch.stack(losses).mean()

    def herpn_range_stats(self):
        stats = {}
        for name, module in self.named_modules():
            if isinstance(module, ProgressiveHerPNActivation):
                stats[name] = module.range_stats()
        return stats

    def herpn_range_summary(self):
        stats = list(self.herpn_range_stats().values())
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

    @torch.no_grad()
    def fold_herpn_for_inference(self):
        """Replace fully converted wrappers by their exact Ax^2+Bx+C form."""
        if self.training:
            raise RuntimeError('Call eval() before folding HerPN activations')
        if float(self.herpn_progress.item()) < float(len(_HERPN_STAGE_NAMES)):
            raise RuntimeError('All five stages must be converted before folding')

        def replace(module):
            for name, child in list(module.named_children()):
                if isinstance(child, ProgressiveHerPNActivation):
                    setattr(module, name, child.folded())
                else:
                    replace(child)

        replace(self)
        return self

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

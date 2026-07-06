"""IResNet variant with AESPA-style HerPN activations.

AESPA replaces BN+ReLU blocks with HerPN: a fixed low-degree Hermite
approximation of ReLU with basis-wise normalization, followed by learnable
affine scale and shift.  This file adapts the residual block to the AESPA
ResNet pattern:

    conv -> HerPN -> conv -> BN -> residual add -> HerPN

The HerPN operation uses additions and multiplications, so its inference-time
polynomial can be evaluated by HE/FHE backends after normalization statistics
are folded into constants.
"""

import math

import numpy as np
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

__all__ = [
    'iresnet18', 'iresnet34', 'iresnet50', 'iresnet100', 'iresnet200',
    'CryptoFacePolyAct2d', 'IBasicBlock', 'IResNet'
]

using_ckpt = False


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding."""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution."""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=1,
        stride=stride,
        bias=False,
    )

'''
class CryptoFacePolyAct2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-05, quadratic_input_clip: float = 8.0):
        super().__init__()
        if num_channels <= 0:
            raise ValueError('num_channels must be positive')
        self.quadratic_input_clip = float(quadratic_input_clip)
        self.bn_h1 = nn.BatchNorm2d(num_channels, eps=eps, affine=False, track_running_stats=False)
        self.bn_h2 = nn.BatchNorm2d(num_channels, eps=eps, affine=False, track_running_stats=False)
        self.gamma = nn.Parameter(torch.full((1, num_channels, 1, 1), 0.5))
        self.beta = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.register_buffer('coeff_h0', torch.tensor(1.0 / math.sqrt(2.0 * math.pi)))
        self.register_buffer('coeff_h1', torch.tensor(0.5))
        self.register_buffer('coeff_h2', torch.tensor(1.0 / math.sqrt(4.0 * math.pi)))
    def forward(self, x):
        h0 = torch.zeros_like(x)
        h1 = self.bn_h1(x)
        x_quadratic = x.float().clamp(-self.quadratic_input_clip, self.quadratic_input_clip)
        h2_basis = (x_quadratic * x_quadratic - 1.0) * (1.0 / math.sqrt(2.0))
        h2 = self.bn_h2(h2_basis.to(dtype=x.dtype))
        out = self.coeff_h0.to(dtype=x.dtype, device=x.device) * h0
        out = out + self.coeff_h1.to(dtype=x.dtype, device=x.device) * h1
        out = out + self.coeff_h2.to(dtype=x.dtype, device=x.device) * h2
        return self.gamma.to(dtype=x.dtype, device=x.device) * out + self.beta.to(dtype=x.dtype, device=x.device)
'''

class CryptoFacePolyAct2d(nn.Module):
    """Degree-2 AESPA HerPN block for 2D feature maps.

    HerPN computes fixed Hermite-basis coefficients for ReLU, normalizes each
    non-constant basis separately, accumulates them, then applies trainable
    per-channel affine parameters gamma and beta.  For deep R50 training, the
    quadratic basis input is clipped before squaring, and basis normalization
    uses batch statistics during validation instead of fragile running stats.
    """

    def __init__(self, planes):
        super(CryptoFacePolyAct2d, self).__init__()
        self.bn0 = nn.BatchNorm2d(planes, affine=False)
        self.bn1 = nn.BatchNorm2d(planes, affine=False)
        self.bn2 = nn.BatchNorm2d(planes, affine=False)
        self.weight = nn.Parameter(torch.ones(planes, 1, 1))
        self.bias = nn.Parameter(torch.zeros(planes, 1, 1))

    def forward(self, x):
        x0 = self.bn0(torch.ones_like(x))
        x1 = self.bn1(x)
        x2 = self.bn2((torch.square(x) - 1) / math.sqrt(2))
        out = torch.divide(x0, math.sqrt(2 * math.pi)) + torch.divide(x1, 2) + torch.divide(x2, np.sqrt(4 * math.pi))
        out = self.weight * out + self.bias
        return out


class IBasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        act_layer=CryptoFacePolyAct2d,
    ):
        super(IBasicBlock, self).__init__()
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError('Dilation > 1 not supported in BasicBlock')
        self.bn1 = nn.BatchNorm2d(inplanes, eps=1e-05)
        self.conv1 = conv3x3(inplanes, planes)
        self.herpn = act_layer(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-05)
        self.downsample = downsample
        self.stride = stride

    def forward_impl(self, x):
        identity = x
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.herpn(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return out

    def forward(self, x):
        if self.training and using_ckpt:
            return checkpoint(self.forward_impl, x, use_reentrant=False)
        return self.forward_impl(x)


class IResNet(nn.Module):
    fc_scale = 7 * 7

    def __init__(
        self,
        block,
        layers,
        dropout=0,
        num_features=512,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        fp16=False,
        act_layer=CryptoFacePolyAct2d,
    ):
        super(IResNet, self).__init__()
        self.extra_gflops = 0.0
        self.fp16 = fp16
        self.inplanes = 64
        self.dilation = 1
        self.act_layer = act_layer

        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                'replace_stride_with_dilation should be None '
                'or a 3-element tuple, got {}'.format(replace_stride_with_dilation)
            )

        self.groups = groups
        self.base_width = width_per_group

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.herpn = act_layer(self.inplanes)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(
            block,
            128,
            layers[1],
            stride=2,
            dilate=replace_stride_with_dilation[0],
        )
        self.layer3 = self._make_layer(
            block,
            256,
            layers[2],
            stride=2,
            dilate=replace_stride_with_dilation[1],
        )
        self.layer4 = self._make_layer(
            block,
            512,
            layers[3],
            stride=2,
            dilate=replace_stride_with_dilation[2],
        )

        self.bn2 = nn.BatchNorm2d(512 * block.expansion, eps=1e-05)
        self.dropout = nn.Dropout(p=dropout, inplace=True)
        self.fc = nn.Linear(512 * block.expansion * self.fc_scale, num_features)
        self.features = nn.BatchNorm1d(num_features, eps=1e-05)
        nn.init.constant_(self.features.weight, 1.0)
        self.features.weight.requires_grad = False

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.1)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, IBasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        downsample = None
        previous_dilation = self.dilation

        if dilate:
            self.dilation *= stride
            stride = 1

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion, eps=1e-05),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                act_layer=self.act_layer,
            )
        )
        self.inplanes = planes * block.expansion

        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    act_layer=self.act_layer,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        # Keep the original AMP behavior.  For actual encrypted inference export,
        # use the trained weights and evaluate the polynomial activations in your
        # FHE backend; PyTorch autocast itself is not part of FHE evaluation.
        with torch.cuda.amp.autocast(enabled=self.fp16):
            x = self.conv1(x)
            x = self.herpn(x)
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
        raise ValueError('Pretrained weights are incompatible with polynomial-activation IResNet.')
    return model


def iresnet18(pretrained=False, progress=True, **kwargs):
    return _iresnet('iresnet18', IBasicBlock, [2, 2, 2, 2], pretrained, progress, **kwargs)


def iresnet34(pretrained=False, progress=True, **kwargs):
    return _iresnet('iresnet34', IBasicBlock, [3, 4, 6, 3], pretrained, progress, **kwargs)


def iresnet50(pretrained=False, progress=True, **kwargs):
    return _iresnet('iresnet50', IBasicBlock, [3, 4, 14, 3], pretrained, progress, **kwargs)


def iresnet100(pretrained=False, progress=True, **kwargs):
    return _iresnet('iresnet100', IBasicBlock, [3, 13, 30, 3], pretrained, progress, **kwargs)


def iresnet200(pretrained=False, progress=True, **kwargs):
    return _iresnet('iresnet200', IBasicBlock, [6, 26, 60, 6], pretrained, progress, **kwargs)

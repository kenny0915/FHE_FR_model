"""
MobileFaceNet variant with FHE-friendly polynomial activations.

This version removes all PReLU/ReLU-style non-polynomial activations and
replaces them with a CryptoFace-style shifted degree-2 polynomial activation:

    y = x^2 + alpha * x + beta

CryptoFace describes replacing ReLU with low-degree polynomial activations
(HerPN/AESPA) and using a shifted monic quadratic form where the leading
coefficient can be folded into the following convolution for lower
multiplicative depth during FHE inference.
"""

import torch
import torch.nn as nn
from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, Sequential, Module


class Flatten(Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ShiftedPolynomialActivation(Module):
    """FHE-compatible shifted degree-2 polynomial activation.

    Computes:
        y = x^2 + alpha * x + beta

    The leading x^2 coefficient is fixed to 1, matching the shifted HerPN
    form used in CryptoFaceNet after folding the quadratic coefficient into
    adjacent linear/convolution weights. alpha and beta are per-channel
    trainable parameters by default.
    """

    def __init__(self, num_parameters, init_alpha=1.0, init_beta=0.0, learnable=True):
        super().__init__()
        shape = (1, num_parameters, 1, 1)
        alpha = torch.full(shape, float(init_alpha))
        beta = torch.full(shape, float(init_beta))

        if learnable:
            self.alpha = nn.Parameter(alpha)
            self.beta = nn.Parameter(beta)
        else:
            self.register_buffer("alpha", alpha)
            self.register_buffer("beta", beta)

    def forward(self, x):
        return x.square() + self.alpha * x + self.beta


class ConvBlock(Module):
    def __init__(
        self,
        in_c,
        out_c,
        kernel=(1, 1),
        stride=(1, 1),
        padding=(0, 0),
        groups=1,
        poly_init_alpha=1.0,
        poly_init_beta=0.0,
    ):
        super(ConvBlock, self).__init__()
        self.layers = nn.Sequential(
            Conv2d(in_c, out_c, kernel, groups=groups, stride=stride, padding=padding, bias=False),
            BatchNorm2d(num_features=out_c),
            ShiftedPolynomialActivation(
                out_c,
                init_alpha=poly_init_alpha,
                init_beta=poly_init_beta,
            ),
        )

    def forward(self, x):
        return self.layers(x)


class LinearBlock(Module):
    """Linear convolutional block.

    Conv + BatchNorm is affine/linear at inference after BN folding, so it is
    compatible with FHE arithmetic. No activation is used here.
    """

    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super(LinearBlock, self).__init__()
        self.layers = nn.Sequential(
            Conv2d(in_c, out_c, kernel, stride, padding, groups=groups, bias=False),
            BatchNorm2d(num_features=out_c),
        )

    def forward(self, x):
        return self.layers(x)


class DepthWise(Module):
    def __init__(self, in_c, out_c, residual=False, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=1):
        super(DepthWise, self).__init__()
        self.residual = residual
        self.layers = nn.Sequential(
            ConvBlock(in_c, out_c=groups, kernel=(1, 1), padding=(0, 0), stride=(1, 1)),
            ConvBlock(groups, groups, groups=groups, kernel=kernel, padding=padding, stride=stride),
            LinearBlock(groups, out_c, kernel=(1, 1), padding=(0, 0), stride=(1, 1)),
        )

    def forward(self, x):
        short_cut = None
        if self.residual:
            short_cut = x
        x = self.layers(x)
        if self.residual:
            output = short_cut + x
        else:
            output = x
        return output


class Residual(Module):
    def __init__(self, c, num_block, groups, kernel=(3, 3), stride=(1, 1), padding=(1, 1)):
        super(Residual, self).__init__()
        modules = []
        for _ in range(num_block):
            modules.append(DepthWise(c, c, True, kernel, stride, padding, groups))
        self.layers = Sequential(*modules)

    def forward(self, x):
        return self.layers(x)


class GDC(Module):
    def __init__(self, embedding_size):
        super(GDC, self).__init__()
        self.layers = nn.Sequential(
            LinearBlock(512, 512, groups=512, kernel=(7, 7), stride=(1, 1), padding=(0, 0)),
            Flatten(),
            Linear(512, embedding_size, bias=False),
            BatchNorm1d(embedding_size),
        )

    def forward(self, x):
        return self.layers(x)


class MobileFaceNet(Module):
    def __init__(self, fp16=False, num_features=512, blocks=(1, 4, 6, 2), scale=2):
        super(MobileFaceNet, self).__init__()
        self.scale = scale
        self.fp16 = fp16
        self.layers = nn.ModuleList()
        self.layers.append(
            ConvBlock(3, 64 * self.scale, kernel=(3, 3), stride=(2, 2), padding=(1, 1))
        )
        if blocks[0] == 1:
            self.layers.append(
                ConvBlock(
                    64 * self.scale,
                    64 * self.scale,
                    kernel=(3, 3),
                    stride=(1, 1),
                    padding=(1, 1),
                    groups=64,
                )
            )
        else:
            self.layers.append(
                Residual(
                    64 * self.scale,
                    num_block=blocks[0],
                    groups=128,
                    kernel=(3, 3),
                    stride=(1, 1),
                    padding=(1, 1),
                ),
            )

        self.layers.extend(
            [
                DepthWise(64 * self.scale, 64 * self.scale, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=128),
                Residual(64 * self.scale, num_block=blocks[1], groups=128, kernel=(3, 3), stride=(1, 1), padding=(1, 1)),
                DepthWise(64 * self.scale, 128 * self.scale, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=256),
                Residual(128 * self.scale, num_block=blocks[2], groups=256, kernel=(3, 3), stride=(1, 1), padding=(1, 1)),
                DepthWise(128 * self.scale, 128 * self.scale, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=512),
                Residual(128 * self.scale, num_block=blocks[3], groups=256, kernel=(3, 3), stride=(1, 1), padding=(1, 1)),
            ]
        )

        self.conv_sep = ConvBlock(128 * self.scale, 512, kernel=(1, 1), stride=(1, 1), padding=(0, 0))
        self.features = GDC(num_features)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # There is no exact Kaiming setting for a polynomial activation.
                # "linear" avoids assuming a ReLU/PReLU nonlinearity.
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="linear")
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="linear")
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        # Keep the original fp16 option for CUDA inference/training, but avoid
        # enabling CPU autocast accidentally.
        autocast_enabled = bool(self.fp16 and x.is_cuda)
        with torch.amp.autocast(device_type="cuda", enabled=autocast_enabled):
            for func in self.layers:
                x = func(x)
        x = self.conv_sep(x.float() if self.fp16 else x)
        x = self.features(x)
        return x


def get_mbf(fp16, num_features, blocks=(1, 4, 6, 2), scale=2):
    return MobileFaceNet(fp16, num_features, blocks, scale=scale)


def get_mbf_large(fp16, num_features, blocks=(2, 8, 12, 4), scale=4):
    return MobileFaceNet(fp16, num_features, blocks, scale=scale)

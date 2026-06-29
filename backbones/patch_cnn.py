import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class HerPN(nn.Module):
    def __init__(self, planes):
        super(HerPN, self).__init__()
        self.bn0 = nn.BatchNorm2d(planes, affine=False)
        self.bn1 = nn.BatchNorm2d(planes, affine=False)
        self.bn2 = nn.BatchNorm2d(planes, affine=False)
        self.weight = nn.Parameter(torch.ones(planes, 1, 1))
        self.bias = nn.Parameter(torch.zeros(planes, 1, 1))

    def forward(self, x):
        x0 = self.bn0(torch.ones_like(x))
        x1 = self.bn1(x)
        x2 = self.bn2((torch.square(x) - 1) / math.sqrt(2))
        out = x0 / math.sqrt(2 * math.pi) + x1 / 2 + x2 / math.sqrt(4 * math.pi)
        out = self.weight * out + self.bias
        return out


class HerPNConv(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(HerPNConv, self).__init__()
        self.herpn1 = HerPN(in_planes)
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.herpn2 = HerPN(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        x = self.herpn1(x)
        out = self.conv1(x)
        out = self.herpn2(out)
        out = self.conv2(out)
        out += self.shortcut(x)
        return out

    @torch.no_grad()
    def fuse(self):
        self._fuse_first_herpn()
        self._fuse_second_herpn()

        weight1 = self.conv1.weight * self.a2
        weight2 = self.conv2.weight * self.b2
        self.weight1 = nn.Parameter(weight1)
        self.weight2 = nn.Parameter(weight2)
        self.a1 = nn.Parameter(self.a1 / self.a2)
        self.a0 = nn.Parameter(self.a0 / self.a2)
        self.b1 = nn.Parameter(self.b1 / self.b2)
        self.b0 = nn.Parameter(self.b0 / self.b2)

    def _fuse_first_herpn(self):
        self.a2, self.a1, self.a0 = _fuse_herpn(self.herpn1)

    def _fuse_second_herpn(self):
        self.b2, self.b1, self.b0 = _fuse_herpn(self.herpn2)

    def forward_fuse(self, x):
        x = torch.square(x) + self.a1 * x + self.a0
        out = F.conv2d(x, self.weight1, stride=self.conv1.stride, padding=self.conv1.padding)
        out = torch.square(out) + self.b1 * out + self.b0
        out = F.conv2d(
            out, self.weight2, stride=self.conv2.stride, padding=self.conv2.padding
        )
        out += self.shortcut(x * self.a2)
        return out


class HerPNPool(nn.Module):
    def __init__(self, planes, output_size):
        super(HerPNPool, self).__init__()
        self.herpn = HerPN(planes)
        self.pool = nn.AdaptiveAvgPool2d(output_size)

    def forward(self, x):
        x = self.herpn(x)
        return self.pool(x)

    @torch.no_grad()
    def fuse(self):
        self.a2, self.a1, self.a0 = _fuse_herpn(self.herpn)
        self.a1 = nn.Parameter(self.a1 / self.a2)
        self.a0 = nn.Parameter(self.a0 / self.a2)
        self.a2 = nn.Parameter(self.a2)

    def forward_fuse(self, x):
        out = torch.square(x) + self.a1 * x + self.a0
        out = F.adaptive_avg_pool2d(out, self.pool.output_size) * self.a2
        return out


def _fuse_herpn(herpn):
    m0, v0 = herpn.bn0.running_mean, herpn.bn0.running_var
    m1, v1 = herpn.bn1.running_mean, herpn.bn1.running_var
    m2, v2 = herpn.bn2.running_mean, herpn.bn2.running_var
    g, b = herpn.weight.squeeze(), herpn.bias.squeeze()
    eps = herpn.bn0.eps

    w2 = g / torch.sqrt(8 * math.pi * (v2 + eps))
    w1 = g / (2 * torch.sqrt(v1 + eps))
    w0 = b + g * (
        (1 - m0) / torch.sqrt(2 * math.pi * (v0 + eps))
        - m1 / (2 * torch.sqrt(v1 + eps))
        - (1 + math.sqrt(2) * m2) / torch.sqrt(8 * math.pi * (v2 + eps))
    )
    return (
        w2.unsqueeze(-1).unsqueeze(-1),
        w1.unsqueeze(-1).unsqueeze(-1),
        w0.unsqueeze(-1).unsqueeze(-1),
    )


def initialize_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        if m.weight is not None:
            m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        if m.bias is not None:
            m.bias.data.zero_()


class PatchBackbone(nn.Module):
    def __init__(self, output_size):
        super(PatchBackbone, self).__init__()
        self.conv = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.layers = nn.Sequential(
            HerPNConv(16, 16),
            HerPNConv(16, 32, 2),
            HerPNConv(32, 32),
            HerPNConv(32, 64, 2),
            HerPNConv(64, 64),
        )
        self.herpnpool = HerPNPool(64, output_size=output_size)
        self.flatten = Flatten()
        self.bn = nn.BatchNorm1d(output_size[0] * output_size[1] * 64)

    def forward(self, x):
        out = self.conv(x)
        for layer in self.layers:
            out = layer(out)
        out = self.herpnpool(out)
        out = self.flatten(out)
        out = self.bn(out)
        return out

    @torch.no_grad()
    def fuse(self):
        for layer in self.layers:
            layer.fuse()
        self.herpnpool.fuse()

    def forward_fuse(self, x):
        out = self.conv(x)
        for layer in self.layers:
            out = layer.forward_fuse(out)
        out = self.herpnpool.forward_fuse(out)
        out = self.flatten(out)
        out = self.bn(out)
        return out


class PatchCNN(nn.Module):
    def __init__(
        self,
        input_size=112,
        patch_size=28,
        num_features=256,
        output_size=(2, 2),
        fp16=False,
    ):
        super(PatchCNN, self).__init__()
        if input_size % patch_size != 0:
            raise ValueError(
                f"input_size must be divisible by patch_size: {input_size} vs {patch_size}"
            )

        self.input_size = input_size
        self.patch_size = patch_size
        self.fp16 = fp16
        self.H = input_size // patch_size
        self.W = input_size // patch_size
        self.num_patches = self.H * self.W
        self.patch_dim = output_size[0] * output_size[1] * 64

        self.nets = nn.ModuleList(
            [PatchBackbone(output_size) for _ in range(self.num_patches)]
        )
        self.linear = nn.Linear(self.num_patches * self.patch_dim, num_features)
        self.bn = nn.BatchNorm1d(num_features, affine=False)
        self.jigsaw = nn.Linear(self.patch_dim, self.num_patches)

        self.apply(initialize_weights)

    @torch.no_grad()
    def fuse(self):
        for net in self.nets:
            net.fuse()
        mean = self.bn.running_mean
        var = self.bn.running_var
        eps = self.bn.eps
        weight = self.linear.weight
        bias = self.linear.bias
        weight = (weight.T / torch.sqrt(var + eps)).T
        bias = (bias - mean) / torch.sqrt(var + eps)
        self.bias = nn.Parameter(bias / self.num_patches)
        self.weights = nn.ParameterList(torch.chunk(weight, self.num_patches, dim=1))

    def _split_patches(self, x):
        batch, channels, height, width = x.shape
        expected = self.input_size
        if height != expected or width != expected:
            raise ValueError(
                f"PatchCNN expects {expected}x{expected} input, got {height}x{width}"
            )
        patch = self.patch_size
        x = x.view(batch, channels, self.H, patch, self.W, patch)
        x = x.permute(2, 4, 0, 1, 3, 5).contiguous()
        return x.view(self.num_patches, batch, channels, patch, patch)

    def _forward_patches(self, x, fuse=False):
        x = self._split_patches(x)
        if x.is_cuda:
            streams = [torch.cuda.Stream(device=x.device) for _ in range(self.num_patches)]
            patch_features = [None for _ in range(self.num_patches)]
            for index in range(self.num_patches):
                with torch.cuda.stream(streams[index]):
                    patch_features[index] = self._forward_one_patch(index, x[index], fuse)
            torch.cuda.current_stream(device=x.device).wait_stream(streams[0])
            for stream in streams[1:]:
                torch.cuda.current_stream(device=x.device).wait_stream(stream)
        else:
            patch_features = [
                self._forward_one_patch(index, x[index], fuse)
                for index in range(self.num_patches)
            ]
        out = torch.stack(patch_features, dim=0)
        return out.permute(1, 0, 2).contiguous()

    def _forward_one_patch(self, index, patch, fuse):
        if not fuse:
            return self.nets[index](patch)
        out = self.nets[index].forward_fuse(patch)
        return out @ self.weights[index].T + self.bias

    def forward_fuse(self, x):
        out = self._forward_patches(x, fuse=True)
        return out.sum(dim=1)

    def forward(self, x):
        if self.fp16 and x.is_cuda:
            with torch.amp.autocast("cuda"):
                return self._forward_impl(x)
        return self._forward_impl(x)

    def _forward_impl(self, x):
        out = self._forward_patches(x, fuse=False)
        embedding = out.reshape(out.size(0), -1)
        embedding = self.linear(embedding)
        embedding = self.bn(embedding)

        if not self.training:
            return embedding

        pred = self.jigsaw(out).reshape(-1, self.num_patches)
        target = torch.arange(self.num_patches, device=out.device).repeat(out.size(0))
        return embedding, pred, target


def patch_cnn(**kwargs):
    return PatchCNN(
        input_size=kwargs.get("input_size", 112),
        patch_size=kwargs.get("patch_size", 28),
        num_features=kwargs.get("num_features", 256),
        fp16=kwargs.get("fp16", False),
    )

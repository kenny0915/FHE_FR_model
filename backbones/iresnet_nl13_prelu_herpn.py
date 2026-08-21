"""Progressively convert the thirteen PReLUs retained by NL13 to HerPN.

For a frozen channel-wise slope ``a``, the degree-2 polynomial targets

    PReLU_a(x) = a*x + (1-a)*ReLU(x),  x in [-6, 6].

AESPA basis-wise normalization fits the empirical input distribution, whose
central mass should remain near [-3, 3], while training monitors and penalizes
excursions beyond the [-6, 6] safety interval.  After BatchNorm calibration,
all thirteen activations fold exactly to ``A*x^2 + B*x + C``.  Each activation
therefore adds one ciphertext-square level and no non-polynomial inference
operation.
"""

from .iresnet import IBasicBlock
from .iresnet_nl9_prelu_herpn import IResNet as _ReducedPReLUHerPNIResNet

__all__ = ["IResNet", "iresnet50"]

NL13_ACTIVATION_NAMES = (
    "prelu",
    "layer1.0.prelu",
    "layer1.2.prelu",
    "layer2.0.prelu",
    "layer2.3.prelu",
    "layer3.0.prelu",
    "layer3.3.prelu",
    "layer3.6.prelu",
    "layer3.9.prelu",
    "layer3.13.prelu",
    "layer4.0.prelu",
    "layer4.1.prelu",
    "layer4.2.prelu",
)


class IResNet(_ReducedPReLUHerPNIResNet):
    """NL13 IResNet50 with progressive PReLU-aware HerPN students."""

    arch_config_name = "nl13"
    activation_names = NL13_ACTIVATION_NAMES


def iresnet50(pretrained=False, progress=True, **kwargs):
    del progress
    if pretrained:
        raise ValueError("No bundled pretrained NL13 PReLU-HerPN checkpoint")
    return IResNet(IBasicBlock, [3, 4, 14, 3], **kwargs)

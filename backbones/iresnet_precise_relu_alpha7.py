"""IResNet curriculum ending at the paper's PreciseReLU Alpha7.

The model starts from the more accurate Alpha10 polynomial and performs one
whole-network transition to Alpha7 on the same fixed public interval. Every
trained channel-wise PReLU slope remains in

    a*x + (1-a)*poly_relu(x).

After progress reaches one, the forward graph uses Alpha7 exclusively. Its
two degree-7 components have nonlinear multiplicative depth 7; the training
blend, range penalty, and surrogate backward are plaintext-only machinery.
"""

from .iresnet import IBasicBlock
from .iresnet_precise_relu import (
    IResNet as _CurriculumIResNet,
    ProgressivePrecisePReLU,
)

__all__ = [
    "ProgressivePrecisePReLU",
    "IResNet",
    "iresnet18",
    "iresnet34",
    "iresnet50",
    "iresnet100",
    "iresnet200",
]


class IResNet(_CurriculumIResNet):
    """Alpha10-to-Alpha7 specialization of the precise-ReLU backbone."""

    def __init__(self, *args, precise_relu_input_scale=16.0,
                 precise_relu_target_alphas=(7,),
                 precise_relu_lower_degrees=(),
                 precise_relu_progress=0.0,
                 precise_relu_backward_mode="relu_ste", **kwargs):
        if tuple(precise_relu_target_alphas) != (7,):
            raise ValueError(
                "Alpha7 backbone requires precise_relu_target_alphas=(7,)")
        if tuple(precise_relu_lower_degrees):
            raise ValueError(
                "Alpha7 backbone does not use lower-degree students")
        super().__init__(
            *args,
            precise_relu_input_scale=precise_relu_input_scale,
            precise_relu_target_alphas=(7,),
            precise_relu_lower_degrees=(),
            precise_relu_progress=precise_relu_progress,
            precise_relu_backward_mode=precise_relu_backward_mode,
            **kwargs
        )


def _iresnet(blocks, pretrained, **kwargs):
    model = IResNet(IBasicBlock, blocks, **kwargs)
    if pretrained:
        raise ValueError("No bundled pretrained Alpha7 checkpoint")
    return model


def iresnet18(pretrained=False, progress=True, **kwargs):
    return _iresnet([2, 2, 2, 2], pretrained, **kwargs)


def iresnet34(pretrained=False, progress=True, **kwargs):
    return _iresnet([3, 4, 6, 3], pretrained, **kwargs)


def iresnet50(pretrained=False, progress=True, **kwargs):
    return _iresnet([3, 4, 14, 3], pretrained, **kwargs)


def iresnet100(pretrained=False, progress=True, **kwargs):
    return _iresnet([3, 13, 30, 3], pretrained, **kwargs)


def iresnet200(pretrained=False, progress=True, **kwargs):
    return _iresnet([6, 26, 60, 6], pretrained, **kwargs)

"""Progressively convert only the nine PReLUs retained by NL9 to HerPN.

For a frozen channel-wise PReLU slope ``a``, the approximation target is

    PReLU_a(x) = a*x + (1-a)*ReLU(x),  x in [-6, 6].

The polynomial student keeps the exact linear PReLU component and replaces
only ReLU with AESPA's basis-normalized degree-2 Hermite activation:

    student_a(x) = a*x + (1-a)*HerPN_ReLU(x).

The interval is a monitored training-time safety range.  HerPN is fitted with
the empirical activation distribution inside that range rather than as a
uniform minimax approximation.  Once its BatchNorm statistics are calibrated,
each student folds exactly to ``A*x^2 + B*x + C``.  Thus the encrypted graph
contains nine ciphertext squares and no PReLU, ReLU, or data-dependent branch.
"""

import torch
from torch import nn

from .iresnet import IBasicBlock, IResNet as _ReducedIResNet
from .iresnet_prelu_herpn import PReLUHerPNActivation

__all__ = ["IResNet", "iresnet50"]

NL9_ACTIVATION_NAMES = (
    "prelu",
    "layer1.0.prelu",
    "layer2.0.prelu",
    "layer3.0.prelu",
    "layer3.3.prelu",
    "layer3.9.prelu",
    "layer3.13.prelu",
    "layer4.0.prelu",
    "layer4.2.prelu",
)

_STAGE_INDEX = {
    "prelu": 0,
    "layer1": 1,
    "layer2": 2,
    "layer3": 3,
    "layer4": 4,
}


def _parent_and_child(module, qualified_name):
    parts = qualified_name.split(".")
    parent = module
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


class IResNet(_ReducedIResNet):
    """NL9 IResNet50 with nine progressive PReLU-aware HerPN students."""

    def __init__(self, *args, arch_config="nl9", activation_mask=None,
                 herpn_range_limit=6.0, herpn_bn_eps=1e-4,
                 herpn_progress=0.0, prelu_herpn_distill_eps=1e-4,
                 **kwargs):
        if str(arch_config) != "nl9":
            raise ValueError(
                "NL9 PReLU-HerPN requires arch_config='nl9', got {!r}".format(
                    arch_config))
        if activation_mask is not None:
            raise ValueError(
                "NL9 PReLU-HerPN uses the fixed nl9 activation mask")
        if herpn_range_limit <= 0.0:
            raise ValueError("herpn_range_limit must be positive")
        if herpn_bn_eps <= 0.0:
            raise ValueError("herpn_bn_eps must be positive")
        if prelu_herpn_distill_eps <= 0.0:
            raise ValueError("prelu_herpn_distill_eps must be positive")

        super().__init__(
            *args,
            arch_config="nl9",
            activation_mask=None,
            **kwargs,
        )
        self.herpn_range_limit = float(herpn_range_limit)
        self.herpn_bn_eps = float(herpn_bn_eps)
        self.prelu_herpn_distill_eps = float(prelu_herpn_distill_eps)
        self.register_buffer(
            "herpn_progress",
            torch.tensor(float(herpn_progress), dtype=torch.float32),
            persistent=False,
        )
        self._replace_retained_prelus()
        self.set_herpn_progress(herpn_progress)

    def _replace_retained_prelus(self):
        actual = {
            name for name, module in self.named_modules()
            if isinstance(module, nn.PReLU)
        }
        expected = set(NL9_ACTIVATION_NAMES)
        if actual != expected:
            raise RuntimeError(
                "NL9 PReLU locations changed; missing={}, extra={}".format(
                    sorted(expected - actual), sorted(actual - expected)))

        for activation_name in NL9_ACTIVATION_NAMES:
            parent, child_name = _parent_and_child(self, activation_name)
            original = getattr(parent, child_name)
            stage_name = (
                "prelu" if activation_name == "prelu"
                else activation_name.split(".", 1)[0]
            )
            replacement = PReLUHerPNActivation(
                channels=original.weight.numel(),
                range_limit=self.herpn_range_limit,
                bn_eps=self.herpn_bn_eps,
                distill_eps=self.prelu_herpn_distill_eps,
                stage_index=_STAGE_INDEX[stage_name],
                blend=0.0,
            )
            with torch.no_grad():
                replacement.prelu.weight.copy_(original.weight)
            setattr(parent, child_name, replacement)

    def progressive_activations(self):
        return [
            module for module in self.modules()
            if isinstance(module, PReLUHerPNActivation)
        ]

    def named_progressive_activations(self):
        return [
            (name, module) for name, module in self.named_modules()
            if isinstance(module, PReLUHerPNActivation)
        ]

    def set_herpn_progress(self, progress):
        """Set coarse five-stage progress; singleton schedules use blends."""
        progress = min(max(float(progress), 0.0), 5.0)
        self.herpn_progress.fill_(progress)
        for activation in self.progressive_activations():
            activation.set_blend(
                min(max(progress - activation.stage_index, 0.0), 1.0))

    def set_herpn_blends(self, blends):
        """Set the nine activation blends used by forward-order conversion."""
        activations = dict(self.named_progressive_activations())
        unknown = sorted(set(blends).difference(activations))
        if unknown:
            raise ValueError(
                "Unknown NL9 PReLU-HerPN activations: {}".format(unknown))
        for name, activation in activations.items():
            activation.set_blend(float(blends.get(name, 0.0)))
        converted_fraction = sum(
            activation._blend for activation in activations.values()
        ) / len(activations)
        self.herpn_progress.fill_(5.0 * converted_fraction)

    def herpn_range_penalty(self):
        penalties = [
            activation.range_penalty()
            for activation in self.progressive_activations()
            if activation.range_penalty() is not None
        ]
        if not penalties:
            return next(self.parameters()).new_zeros(())
        return torch.stack(penalties).mean()

    def herpn_distillation_loss(self):
        losses = [
            activation.distillation_loss()
            for activation in self.progressive_activations()
            if activation.distillation_loss() is not None
        ]
        if not losses:
            return next(self.parameters()).new_zeros(())
        return torch.stack(losses).mean()

    def herpn_range_stats(self):
        return {
            name: module.range_stats()
            for name, module in self.named_progressive_activations()
        }

    def herpn_range_summary(self):
        stats = list(self.herpn_range_stats().values())
        absmax = [
            item["absmax"] for item in stats if item["absmax"] is not None
        ]
        outside = [
            item["outside_fraction"] for item in stats
            if item["outside_fraction"] is not None
        ]
        zero = next(self.parameters()).new_zeros(())
        return {
            "input_absmax": torch.stack(absmax).amax() if absmax else zero,
            "outside_fraction": (
                torch.stack(outside).mean() if outside else zero),
        }

    def begin_batchnorm_recalibration(self, reset=True):
        """Update only BatchNorm statistics after each completed singleton."""
        batchnorm_state = [
            (module, module.training, module.momentum)
            for module in self.modules()
            if isinstance(module, nn.modules.batchnorm._BatchNorm)
        ]
        state = {
            "model_training": self.training,
            "batchnorm": batchnorm_state,
        }
        self.eval()
        for module, _, _ in batchnorm_state:
            if reset:
                module.reset_running_stats()
            module.momentum = None
            module.train()
        return state

    def end_batchnorm_recalibration(self, state):
        self.train(state["model_training"])
        for module, was_training, momentum in state["batchnorm"]:
            module.momentum = momentum
            module.train(was_training)

    @torch.no_grad()
    def fold_herpn_for_inference(self):
        """Replace all nine converted wrappers by exact quadratics."""
        if self.training:
            raise RuntimeError("Call eval() before folding NL9 PReLU-HerPN")
        activations = self.progressive_activations()
        if any(activation._blend < 1.0 for activation in activations):
            raise RuntimeError(
                "All nine NL9 activations must be converted before folding")

        def replace(module):
            for name, child in list(module.named_children()):
                if isinstance(child, PReLUHerPNActivation):
                    setattr(module, name, child.folded())
                else:
                    replace(child)

        replace(self)
        return self


def iresnet50(pretrained=False, progress=True, **kwargs):
    del progress
    if pretrained:
        raise ValueError("No bundled pretrained NL9 PReLU-HerPN checkpoint")
    return IResNet(IBasicBlock, [3, 4, 14, 3], **kwargs)

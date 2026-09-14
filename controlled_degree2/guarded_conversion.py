"""Training-only finite-prefix routing; exported inference remains unchanged.

Every row contributes either its full backbone features or a finite prefix
repair loss. Escaping rows are retained for training replay, never evaluated
with replacement embeddings. Only the training caller sees the dummy feature
when an entire rank batch needs prefix repair.
"""
import torch
from torch import nn

from controlled_degree2.recipe_a_recovery import finite_prefix_loss
from controlled_degree2.model import quadratic_modules


class GuardedConversion(nn.Module):
    def __init__(self, backbone, guard=4., target=2., feature_dim=512):
        super().__init__()
        if not 0 < target < guard:
            raise ValueError('require 0 < target < guard')
        self.backbone = backbone
        self.guard, self.target, self.feature_dim = guard, target, feature_dim

    def forward(self, images):
        if not self.training:
            raise RuntimeError('guarded routing is training-only; evaluate the underlying backbone')
        pending = torch.arange(len(images), device=images.device)
        groups = []
        scores = images.new_zeros(len(images))
        # Probe before any unsafe square, preserving the exact current phase.
        with torch.no_grad():
            while len(pending):
                _, site, ratios = finite_prefix_loss(
                    self.backbone, images[pending], 25, self.guard, self.target,
                    preserve_phase=True)
                if site is None:
                    break
                escaping = ratios > self.guard
                groups.append(pending[escaping])
                scores[pending[escaping]] = ratios[escaping]
                pending = pending[~escaping]
        repair = images.new_zeros((), dtype=torch.float64)
        for rows in groups:
            loss, site, _ = finite_prefix_loss(
                self.backbone, images[rows], 25, self.guard, self.target,
                preserve_phase=True)
            if site is None or not torch.isfinite(loss):
                raise FloatingPointError('prefix routing changed between probe and repair')
            repair = repair + torch.log1p(loss / self.guard**2) * (len(rows)/len(images))
        if len(pending):
            features = self.backbone(images[pending])
            scores[pending] = torch.stack([m.last_sample_ratio for m in quadratic_modules(self.backbone)]).amax(0)
            if not torch.isfinite(features).all() or not torch.isfinite(features.norm(dim=1)).all():
                raise FloatingPointError('guarded full rows produced nonfinite features/norms')
        else:
            # Keeps an all-repair rank compatible with the separate DDP head;
            # the caller must provide an all-false identity mask for this row.
            features = (repair.float()*0).expand(1, self.feature_dim)
        return features, pending, repair, scores

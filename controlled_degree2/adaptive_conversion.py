"""Unclipped hybrid activations and finite/quality gates for conversion pilots.

The hybrid exists only during conversion. A deployment candidate requires
alpha=1 at every site and a separate full polynomial-export evaluation.
"""
from dataclasses import dataclass
import math
import torch
from torch.nn import functional as F
from controlled_degree2.model import DirectQuadratic


class ConversionQuadratic(DirectQuadratic):
    def forward(self, x):
        # Unlike the legacy training-only blend, eval probes must evaluate
        # exactly the current hybrid, including unopened PReLU suffixes.
        if self.alpha == 0:
            return F.prelu(x, self.slope)
        c, b, a = self.coeffs.T.reshape(3, *self._view(x))
        polynomial = c+b*x+a*(x*x)
        if self.alpha == 1:
            return polynomial
        return self.alpha*polynomial+(1-self.alpha)*F.prelu(x, self.slope)


@dataclass
class ConversionGate:
    baseline_kd: float
    baseline_ratio: float
    patience: int = 2
    kd_tolerance: float = .02
    streak: int = 0

    def observe(self, report):
        finite = (report['nonfinite'] == 0 and
                  all(math.isfinite(report[k]) for k in ('kd', 'max_ratio')))
        passed = (finite and report['kd'] <= self.baseline_kd+self.kd_tolerance
                  and report['max_ratio'] <= max(4., 1.25*self.baseline_ratio))
        self.streak = self.streak+1 if passed else 0
        return self.streak >= self.patience


def tail_penalty(x, radius):
    """Differentiable mean and per-image worst excess; no output sanitization."""
    relative = x.abs()/radius.reshape(1, -1, 1, 1)
    if not torch.isfinite(relative).all() or relative.detach().max() > 32:
        raise FloatingPointError('activation escaped finite conversion guard')
    excess = (relative-.9).clamp_min(0).square().flatten(1)
    return excess.mean()+.01*excess.amax(1).mean(), relative.detach().flatten(1).amax(1)

"""Training-only continuation controls; exported coefficients remain scalar.

Approximation target/interval are inherited from the MS1MV3 pooled PReLU fit.
Neither coefficient projection nor the clamp curriculum executes in eval.
"""
import math
import torch
from controlled_degree2.model import quadratic_modules


def apply_unclipping(student, epoch, duration):
    modules = list(quadratic_modules(student))
    opened = len(modules) if duration == 0 else min(len(modules), int(len(modules)*epoch/duration))
    for index, module in enumerate(modules):
        module.alpha = 1.
        module.clip = index >= opened
        module.clip_eval = False
    return opened < len(modules)


@torch.no_grad()
def project_coefficients(student, curvature_cap):
    """Box projection in dimensionless (c/R, b, a*R) coordinates."""
    if curvature_cap == 0:
        return
    if not math.isfinite(curvature_cap) or curvature_cap <= 0:
        raise ValueError('curvature cap must be finite and positive')
    for module in quadratic_modules(student):
        if module.coeffs.shape != (1, 3) or module.lam_fit.numel() != 1:
            raise ValueError('projection requires layer-shared coefficients and radius')
        radius = module.lam_fit.item()
        module.coeffs[:, 0].clamp_(-.5*radius, .5*radius)
        module.coeffs[:, 1].clamp_(-1.5, 1.5)
        # Nonzero curvature at every site, with original curvature sign.
        a = module.coeffs[:, 2]
        sign = torch.where(a < 0, -1., 1.)
        a.copy_(sign*a.abs().clamp(1e-6/radius, curvature_cap/radius))

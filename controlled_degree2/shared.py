"""One trainable (c, b, a) per PReLU site; no channel-dependent polynomial.

Fit the original channel PReLU curves jointly over pooled MS1MV3 activation
histograms on [-radius, radius]. Symmetric histogram weighting and a 5%
uniform edge component give a degree-two least-squares approximation.
Only training uses the original slopes. The default evaluation graph uses
c + b*x + a*x*x, one ciphertext square per activation (25 serial sites).
The separately named bounded variant retains fixed input clamps in evaluation;
those comparisons are non-polynomial and must be accounted for under FHE.
"""
import torch
from torch import nn
from controlled_degree2.model import DirectQuadratic, _set_module, prelu_names


class SharedQuadratic(DirectQuadratic):
    def __init__(self, slope, radius=6., coefficients=None, name=''):
        slope = torch.as_tensor(slope).detach().float().reshape(-1)
        super().__init__(1, lam_fit=radius, lam_reg=radius*.6,
                         coeffs=coefficients, slope=float(slope.mean()), name=name)
        self.slope = slope.clone()  # training-only PReLU blend, never polynomial coefficients
        self.coeffs.requires_grad_(True)


def fit_shared(counts, centers, slopes, radius, initialization='fit'):
    counts, x, slopes = counts.double(), centers.double(), slopes.double()
    keep = x <= radius
    x, counts = x[keep], counts[:, keep]
    if x.numel() < 2 or counts.sum() <= 0:
        raise ValueError('insufficient histogram support for quadratic fit')
    probabilities = counts / counts.sum()
    even = .5*(1-slopes)
    linear = .5*(1+slopes)
    # Weighted normal equations for c+a*x^2 and b*x, including synthetic
    # uniform samples inside the same fixed interval (equal channel weighting).
    grid = torch.linspace(0, float(radius), 257, device=x.device, dtype=x.dtype)
    pooled = probabilities.sum(0)
    basis = torch.stack((torch.ones_like(x), x.square()), 1)
    edge_basis = torch.stack((torch.ones_like(grid), grid.square()), 1)
    matrix = .95*(basis.T @ (pooled[:, None]*basis)) + .05*(edge_basis.T @ edge_basis)/len(grid)
    target = (probabilities*even[:, None]).sum(0)*x
    rhs = .95*(basis.T @ target) + .05*(edge_basis.T @ (even.mean()*grid))/len(grid)
    ca = torch.linalg.solve(matrix, rhs)
    b = (.95*(probabilities*linear[:, None]*x.square()).sum()+.05*linear.mean()*grid.square().mean()) / (.95*(pooled*x.square()).sum()+.05*grid.square().mean())
    coeffs = torch.stack((ca[0], b, ca[1])).float().reshape(1, 3)
    if initialization == 'near_linear':
        coeffs[0, 0] = 0.
        coeffs[0, 2] = 1e-3 / float(radius)
    elif initialization != 'fit':
        raise ValueError(initialization)
    return coeffs


def replace_shared(model, calibration):
    names = prelu_names(model)
    if set(names) != set(calibration):
        raise ValueError('calibration must match every PReLU site exactly')
    for name in names:
        original = model.get_submodule(name)
        entry = calibration[name]
        module = SharedQuadratic(original.weight, entry['radius'], entry['coefficients'], name)
        module.lam_reg.fill_(entry['radius']*entry.get('reg_ratio', .6))
        _set_module(model, name, module)
    return names


def build_shared_iresnet50(inference_bound=False, **kwargs):
    from backbones.iresnet import iresnet50
    model = iresnet50(**kwargs)
    for name in prelu_names(model):
        original = model.get_submodule(name)
        module = SharedQuadratic(original.weight, name=name)
        module.clip_eval = bool(inference_bound)
        _set_module(model, name, module)
    return model

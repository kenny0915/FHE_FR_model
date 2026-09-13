"""Export the embedding backbone using only affine maps and quadratics.

BatchNorm inverse standard deviations are calculated once at export, then
stored as constants. L2 normalization, template pooling and scoring belong to
the existing plaintext evaluation protocol, outside this exported backbone.
"""
import copy
import operator
import torch
from torch import nn
from torch.fx import symbolic_trace
from controlled_degree2.shared import SharedQuadratic


class Polynomial(nn.Module):
    def __init__(self, source):
        super().__init__()
        for name, value in zip(('c', 'b', 'a'), source.coeffs.detach().reshape(3)):
            self.register_buffer(name, value.clone())

    def forward(self, x):
        return self.c + self.b*x + self.a*(x*x)


class Affine(nn.Module):
    def __init__(self, bn):
        super().__init__()
        shape = (1, -1, 1, 1) if isinstance(bn, nn.BatchNorm2d) else (1, -1)
        weight = bn.weight.detach() if bn.affine else torch.ones_like(bn.running_mean)
        bias = bn.bias.detach() if bn.affine else torch.zeros_like(bn.running_mean)
        scale = weight * torch.rsqrt(bn.running_var.detach()+bn.eps)
        self.register_buffer('scale', scale.reshape(shape))
        self.register_buffer('offset', (bias-bn.running_mean.detach()*scale).reshape(shape))

    def forward(self, x):
        return x*self.scale+self.offset


def export_graph(model, expected_sites=25):
    if model.training:
        raise ValueError('export requires evaluation mode')
    sites = [m for m in model.modules() if isinstance(m, SharedQuadratic)]
    if len(sites) != expected_sites or any(m.clip_eval or m.coeffs.shape != (1, 3) for m in sites):
        raise ValueError('expected unclipped layer-shared quadratic at every site')
    if any(not torch.isfinite(m.coeffs).all() or m.coeffs[0, 2] == 0 for m in sites):
        raise ValueError('every site must have finite coefficients and nonzero quadratic term')
    result = copy.deepcopy(model).eval()
    for name, module in list(result.named_modules()):
        replacement = None
        if isinstance(module, SharedQuadratic):
            replacement = Polynomial(module)
        elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            replacement = Affine(module)
        elif isinstance(module, nn.Dropout):
            replacement = nn.Identity()
        if replacement is not None:
            parent, _, key = name.rpartition('.')
            setattr(result.get_submodule(parent), key, replacement)
    graph = symbolic_trace(result)
    for node in graph.graph.nodes:
        if node.op in ('placeholder', 'get_attr', 'output'):
            continue
        if node.op == 'call_function' and node.target in (operator.add, operator.mul, torch.flatten):
            continue
        if node.op == 'call_method' and node.target in ('add', 'mul'):
            continue
        if node.op == 'call_module' and isinstance(graph.get_submodule(node.target), (nn.Conv2d, nn.Linear, nn.Identity)):
            continue
        raise ValueError(f'non-polynomial or unrecognized inference operation: {node.op} {node.target}')
    if any(not torch.isfinite(v).all() for v in graph.state_dict().values()):
        raise ValueError('non-finite exported constants')
    return graph, dict(pure_polynomial=True, quadratic_sites=len(sites), coefficients=3*len(sites),
                       operations=['addition', 'multiplication', 'affine convolution/linear', 'reshape'],
                       batchnorm='fixed scale/offset constants', inference_clipping=False,
                       scope='image-to-embedding backbone before plaintext normalization/scoring')

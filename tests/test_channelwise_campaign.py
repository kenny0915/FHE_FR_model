from types import SimpleNamespace

import pytest
import torch
from torch import nn

from controlled_degree2.model import DirectQuadratic
from controlled_degree2.polynomial_export import export_graph
from controlled_degree2.recipe_a import apply_phase, check_resume_policy


def test_sitewise_conversion_has_no_clipping_and_keeps_suffix_prelu():
    model = nn.Sequential(*[DirectQuadratic(2, name='prelu') for _ in range(3)])
    args = SimpleNamespace(head_warmup=1., conversion_epochs=3.,
                           sitewise_unclipped=True, batchnorm_mode='frozen')
    alpha, clipped = apply_phase(model.train(), 2.5, args)
    assert alpha == [1., .5, 0.] and not clipped
    assert all(not m.clip and not m.clip_eval for m in model)
    x = torch.tensor([[[[-3., 2.]], [[-1., 4.]]]])
    torch.testing.assert_close(model[2](x), torch.nn.functional.prelu(x, model[2].slope))
    assert apply_phase(model, 4., args) == ([1., 1., 1.], False)


def test_channelwise_export_preserves_different_coefficients_outside_fit_interval():
    model = nn.Sequential(nn.Conv2d(2, 2, 1), nn.BatchNorm2d(2),
                          DirectQuadratic(2, coeffs=torch.tensor([[.1, .6, .2], [-.2, .8, .05]]))).eval()
    graph, report = export_graph(model, expected_sites=1, coefficient_mode='channelwise')
    x = torch.randn(3, 2, 4, 4)*20
    torch.testing.assert_close(graph(x), model(x))
    assert report['coefficients'] == 6 and report['coefficient_mode'] == 'channelwise'
    assert not report['inference_clipping']
    with pytest.raises(ValueError, match='coefficient mode'):
        export_graph(model, expected_sites=1)
    model[2].coeffs.data[1, 2] = 0
    with pytest.raises(ValueError, match='nonzero'):
        export_graph(model, expected_sites=1, coefficient_mode='channelwise')


def test_trainable_coefficients_receive_finite_gradients_through_conversion():
    q = DirectQuadratic(2).train()
    q.coeffs.requires_grad_(True)
    q.clip = False
    optimizer = torch.optim.SGD([q.coeffs], lr=.01)
    before = q.coeffs.detach().clone()
    x = torch.randn(3, 2, 4, 4)
    for alpha in (0., .5, 1.):
        q.alpha = alpha
        optimizer.zero_grad(set_to_none=True)
        q(x).square().mean().backward()
        assert q.coeffs.grad is not None and torch.isfinite(q.coeffs.grad).all()
        if alpha == 0:
            assert not q.coeffs.grad.any()
        optimizer.step()
    assert not torch.equal(q.coeffs, before)


def test_resume_rejects_changed_channel_policy():
    args = SimpleNamespace(sitewise_unclipped=True, train_channel_coefficients=True)
    with pytest.raises(ValueError, match='sitewise_unclipped'):
        check_resume_policy(dict(sitewise_unclipped=False), args)


def test_ijbc_calibration_updates_finite_prefix_and_restores_unclipped_graph():
    from controlled_degree2.calibrate_ijbc_channelwise import calibration_loss, parameters_for_calibration
    from controlled_degree2.recipe_a_recovery_v2 import sitewise

    class Toy(nn.Module):
        def __init__(self, polynomial):
            super().__init__()
            self.conv = nn.Conv2d(2, 2, 1)
            self.bn = nn.BatchNorm2d(2)
            self.prelu = DirectQuadratic(2, lam_fit=.25, name='prelu') if polynomial else nn.PReLU(2)

        def forward(self, x):
            return self.prelu(self.bn(self.conv(x))).mean((2, 3))

    torch.manual_seed(4)
    model, teacher = Toy(True).eval(), Toy(False).eval().requires_grad_(False)
    sitewise(model)
    params = parameters_for_calibration(model)
    bn_before = model.bn.running_var.clone()
    loss, metrics = calibration_loss(model, teacher, torch.randn(4, 2, 4, 4)*5)
    loss.backward()
    assert metrics['prefix'] > 0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
    assert model.prelu.coeffs.grad.abs().sum() > 0
    assert not model.prelu.clip_eval and not model.prelu.clip
    assert all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules())
    torch.testing.assert_close(model.bn.running_var, bn_before, rtol=0, atol=0)


def test_acceptance_rejects_rounding_partial_audit_and_wrong_checkpoint():
    from controlled_degree2.channelwise_acceptance import assess
    raw = [dict(points={'0.0001': dict(tar_percent=96., actual_far=.0000999)})]
    audit = dict(target='IJBC', source_images=469375, augmented_embeddings=938750,
                 nonfinite_values=0, embedding_nonfinite_rows=0)
    certificate = dict(pure_polynomial=True, inference_clipping=False, quadratic_sites=25,
                       coefficient_mode='channelwise', coefficients=17664, checkpoint_sha256='abc')
    assert assess(raw, audit, certificate, 'abc')['target_met']
    raw[0]['points']['0.0001']['tar_percent'] = 95.9999
    assert not assess(raw, audit, certificate, 'abc')['target_met']
    raw[0]['points']['0.0001']['tar_percent'] = 96.
    for key, value in (('augmented_embeddings', 938748), ('nonfinite_values', 1)):
        assert not assess(raw, dict(audit, **{key:value}), certificate, 'abc')['target_met']
    assert not assess(raw, audit, certificate, 'different')['target_met']

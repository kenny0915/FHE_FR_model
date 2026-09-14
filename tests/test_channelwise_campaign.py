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
                 audited_input_rows=938750, audited_output_rows=938750,
                 nonfinite_values=0, embedding_nonfinite_rows=0)
    certificate = dict(pure_polynomial=True, inference_clipping=False, quadratic_sites=25,
                       coefficient_mode='channelwise', coefficients=17664, checkpoint_sha256='abc')
    assert assess(raw, audit, certificate, 'abc')['target_met']
    raw[0]['points']['0.0001']['tar_percent'] = 95.9999
    assert not assess(raw, audit, certificate, 'abc')['target_met']
    raw[0]['points']['0.0001']['tar_percent'] = 96.
    for key, value in (('augmented_embeddings', 938748), ('nonfinite_values', 1),
                       ('audited_input_rows', 938748), ('audited_output_rows', 0)):
        assert not assess(raw, dict(audit, **{key:value}), certificate, 'abc')['target_met']
    assert not assess(raw, audit, certificate, 'different')['target_met']


def test_audit_counts_actual_full_and_remainder_forwards():
    from eval.finite_audit import FiniteAudit
    model = nn.Sequential(nn.Linear(3, 2)).eval()
    audit = FiniteAudit(model)
    model(torch.randn(4, 3))
    model(torch.randn(1, 3))
    result = audit.result()
    audit.close()
    assert result['boundaries']['.input[0]']['observed_rows'] == 5
    assert result['boundaries']['.output']['observed_rows'] == 5
    assert result['nonfinite_values'] == 0


def test_failure_capture_preserves_exact_batch_and_preupdate_weights(tmp_path):
    import json
    from controlled_degree2.recipe_a import capture_failure
    q = DirectQuadratic(2, name='prelu')
    q.coeffs.requires_grad_(True)
    model = nn.Sequential(q)
    head = nn.Linear(2, 3)
    optimizer = torch.optim.SGD(model.parameters(), lr=.01)
    images = torch.full((2, 2, 2, 2), 1e25)
    q.clip = False
    features = model(images)
    before = q.coeffs.detach().clone()
    capture_failure(tmp_path, model, head, optimizer,
                    dict(calibration={}, metadata={'source':'test'}),
                    SimpleNamespace(teacher='teacher.pt'), 0, 5, 850, 5.34,
                    images, torch.tensor([0, 1]), torch.tensor([True, False]),
                    dict(features=features), {1:features})
    folder = tmp_path/'numerical_failure_e5_s850'
    report = json.loads((folder/'rank0.json').read_text())
    assert report['tensors']['features']['nonfinite'] > 0
    batch = torch.load(folder/'rank0.pt', weights_only=False)
    torch.testing.assert_close(batch['images'], images, rtol=0, atol=0)
    state = torch.load(folder/'state.pt', weights_only=False)
    torch.testing.assert_close(state['state_dict_backbone']['0.coeffs'], before, rtol=0, atol=0)
    torch.testing.assert_close(q.coeffs, before, rtol=0, atol=0)
    assert state['diagnostic_only'] and not state['pure_quadratic']


def test_loss_diagnostics_distinguish_masked_overflow_from_active_overflow():
    from controlled_degree2.replay_numerical_failure import hint_arithmetic
    student = torch.tensor([[1., 1.], [1e25, 1e25]])
    teacher = torch.ones_like(student)
    report = hint_arithmetic(student, teacher, torch.tensor([True, False]))
    assert report['fp32']['legacy_masked_mean'] == 'nan'
    assert report['fp32']['active_mean'] == 0
    assert report['fp32']['nonfinite_active_rows'] == 0
    active = hint_arithmetic(student, teacher, torch.tensor([True, True]))
    assert active['fp32']['nonfinite_active_rows'] == 1
    assert active['fp64']['nonfinite_active_rows'] == 0
    assert isinstance(active['fp64']['active_mean'], float)


def test_identity_distillation_excludes_pathologies_before_squaring():
    from controlled_degree2.recipe_a import recipe_loss
    q = DirectQuadratic(2)
    q.last_penalty = torch.tensor(0.)
    q.last_sample_penalty = torch.zeros(2)
    features = torch.ones(2, 2, requires_grad=True)
    s = torch.tensor([[1., 1.], [1e25, 1e25]], requires_grad=True)
    kd, _ = recipe_loss(features, features.detach(), {1:s}, {1:torch.ones_like(s)},
                        torch.tensor([True, False]), [q], {q.name:2}, .3)
    assert torch.isfinite(kd)
    kd.backward()
    assert torch.isfinite(s.grad).all() and not s.grad[1].any()
    empty, _ = recipe_loss(features, features.detach(), {1:s}, {1:torch.ones_like(s)},
                           torch.tensor([False, False]), [q], {q.name:2}, .3)
    assert empty == 0


def test_pathology_ablation_requires_explicit_resume_revision():
    args = SimpleNamespace(pathological_fraction=0.)
    with pytest.raises(ValueError, match='policy revision'):
        check_resume_policy({}, args)
    args.resume_policy_revision = 'ablate synthetic pathology after captured row-125 overflow'
    check_resume_policy({}, args)

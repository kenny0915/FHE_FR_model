from types import SimpleNamespace
import json
import pytest
import torch
from torch import nn
from controlled_degree2.shared import SharedQuadratic
from controlled_degree2.recipe_a import apply_phase
from controlled_degree2.unclipped_policy import project_coefficients
from controlled_degree2.polynomial_export import export_graph


def model():
    return nn.Sequential(nn.Conv2d(3, 3, 1), nn.BatchNorm2d(3),
                         SharedQuadratic(torch.ones(3)*.25, radius=2., coefficients=[[.2, .6, .1]], name='prelu'))


def test_curriculum_never_enables_eval_clipping():
    q = model()
    args = SimpleNamespace(all_quadratic_start=True, inference_bound=False,
                           batchnorm_mode='frozen', unclipped_continuation=True, unclip_epochs=6.)
    assert apply_phase(q.train(), 0., args)[1]
    assert q[2].clip and not q[2].clip_eval
    assert not apply_phase(q.train(), 6., args)[1]
    assert not q[2].clip and not q[2].clip_eval
    x = torch.full((1, 3, 2, 2), 20.)
    q[2].eval()
    torch.testing.assert_close(q[2](x), .2+.6*x+.1*x*x)


def test_projection_and_export_preserve_scalar_contract():
    q = model()
    with torch.no_grad():
        q[2].coeffs.copy_(torch.tensor([[20., 3., -2.]]))
    project_coefficients(q, .15)
    torch.testing.assert_close(q[2].coeffs, torch.tensor([[1., 1.5, -.075]]))
    q.eval()
    graph, report = export_graph(q, expected_sites=1)
    x = torch.randn(2, 3, 4, 4)*5
    torch.testing.assert_close(q(x), graph(x), atol=2e-5, rtol=1e-5)
    assert report['pure_polynomial'] and report['coefficients'] == 3
    q[2].clip_eval = True
    with pytest.raises(ValueError, match='unclipped'):
        export_graph(q, expected_sites=1)


def test_export_rejects_hidden_nonlinearity():
    q = nn.Sequential(model(), nn.ReLU()).eval()
    with pytest.raises(ValueError, match='non-polynomial'):
        export_graph(q, expected_sites=1)


def test_full_backbone_export():
    from controlled_degree2.shared import build_shared_iresnet50
    graph, report = export_graph(build_shared_iresnet50(dropout=0, fp16=False).eval())
    assert report['quadratic_sites'] == 25
    assert not any(isinstance(m, (nn.PReLU, nn.BatchNorm2d, nn.BatchNorm1d)) for m in graph.modules())


def test_success_uses_strict_far_and_export_audit(tmp_path):
    from controlled_degree2.unclipped_campaign import result
    out = tmp_path/'unclipped_d2'
    out.mkdir()
    raw = out/'ijbc_tar_at_far_raw.json'
    raw.write_text(json.dumps([dict(points={'0.0001': dict(tar_percent=96.559, actual_far=.0001),
                                              '0.1': dict(tar_percent=99.9, actual_far=.1)})]))
    (tmp_path/'finite_audit.json').write_text(json.dumps(dict(nonfinite_values=0, embedding_nonfinite_rows=0)))
    (tmp_path/'polynomial_certificate.json').write_text(json.dumps(dict(pure_polynomial=True, quadratic_sites=25, inference_clipping=False)))
    assert not result(tmp_path)['target_met']
    raw.write_text(json.dumps([dict(points={'0.0001': dict(tar_percent=96.56, actual_far=.0001)})]))
    assert result(tmp_path)['target_met']
    (tmp_path/'finite_audit.json').write_text(json.dumps(dict(nonfinite_values=1, embedding_nonfinite_rows=0)))
    assert not result(tmp_path)['target_met']


def test_allocation_counts_failed_jobs_and_actual_elapsed():
    from controlled_degree2.campaign_accounting import summarize
    report = summarize(['123|FAILED|120|s|e|node|billing=128,gres/gpu=16',
                        '124|RUNNING|60|s|Unknown|node|gres/gpu=1'])
    assert report['allocated_gpu_hours'] == pytest.approx((120*16+60)/3600)
    assert len(report['allocations']) == 2

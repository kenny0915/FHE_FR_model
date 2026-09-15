import torch
from torch.nn import functional as F
from controlled_degree2.adaptive_conversion import ConversionGate, ConversionQuadratic, tail_penalty


def test_gate_requires_consecutive_finite_quality_checks():
    gate = ConversionGate(.01, 2.)
    good = dict(nonfinite=0, kd=.02, max_ratio=3.)
    assert not gate.observe(good)
    assert not gate.observe(dict(good, nonfinite=1))
    assert not gate.observe(good)
    assert gate.observe(good)
    assert not gate.observe(dict(good, kd=.04))
    assert not gate.observe(dict(good, max_ratio=float('inf')))
    assert not gate.observe(dict(good, kd=float('nan')))


def test_probe_and_training_use_same_hybrid_without_clipping():
    q = ConversionQuadratic(channels=1, lam_fit=[1.], lam_reg=[.6],
                            coeffs=torch.tensor([[.1, .6, .2]]), slope=torch.tensor([.2]))
    x = torch.tensor([[[[-3., 2.]]]])
    for alpha in (0., .25, 1.):
        q.alpha = alpha
        expected = alpha*(.1+.6*x+.2*x*x)+(1-alpha)*F.prelu(x, q.slope)
        torch.testing.assert_close(q.train()(x), expected)
        torch.testing.assert_close(q.eval()(x), expected)


def test_tail_penalty_pushes_extremes_towards_range_and_rejects_overflow():
    import pytest
    x = torch.tensor([[[[.2, 2., -3.]]]], requires_grad=True)
    loss, peak = tail_penalty(x, torch.tensor([1.]))
    loss.backward()
    assert x.grad[0, 0, 0, 0] == 0
    assert x.grad[0, 0, 0, 1] > 0 and x.grad[0, 0, 0, 2] < 0
    assert peak.item() == 3.
    for value in (float('inf'), float('nan'), 33.):
        with pytest.raises(FloatingPointError):
            tail_penalty(torch.full((1, 1, 1, 1), value), torch.ones(1))


def test_export_refuses_hybrid_but_accepts_completed_conversion():
    import pytest
    from controlled_degree2.polynomial_export import export_graph
    q = ConversionQuadratic(channels=1, coeffs=torch.tensor([[.1, .6, .2]]))
    model = torch.nn.Sequential(q).eval()
    q.alpha = .75
    with pytest.raises(ValueError, match='incompletely converted'):
        export_graph(model, expected_sites=1, coefficient_mode='channelwise')
    q.alpha = 1.
    graph, certificate = export_graph(model, expected_sites=1, coefficient_mode='channelwise')
    x = torch.tensor([[[[-3., 2.]]]])
    torch.testing.assert_close(graph(x), model(x))
    assert certificate['pure_polynomial']

import torch
from torch import nn
from controlled_degree2.shared import SharedQuadratic, fit_shared, replace_shared, build_shared_iresnet50
from eval.finite_audit import FiniteAudit


def test_shared_coefficients_and_training_only_teacher():
    q = SharedQuadratic(torch.tensor([.1, .8, -.2]), coefficients=[[.2, .5, .1]])
    assert q.coeffs.shape == (1, 3) and q.coeffs.requires_grad
    q.eval()
    x = torch.tensor([[[[2.]], [[2.]], [[2.]]]])
    torch.testing.assert_close(q(x), .2+.5*x+.1*x*x)
    q.train()
    q.alpha = 0.
    torch.testing.assert_close(q(-x), torch.nn.functional.prelu(-x, q.slope))
    q.alpha = 1.
    q.clip = False
    q(x).sum().backward()
    assert torch.isfinite(q.coeffs.grad).all() and q.coeffs.grad.abs().sum() > 0


def test_joint_fit_matches_uniform_prelu_solution():
    x = torch.linspace(0, 3, 257, dtype=torch.float64)
    slopes = torch.tensor([.1, .5], dtype=torch.float64)
    fit = fit_shared(torch.ones(2, 257), x, slopes, 3)
    # Dense discretization of uniform L2 fit |x| ~ 3R/16+15x²/(16R).
    expected = torch.tensor([[.35*9/16, .65, .35*15/48]])
    torch.testing.assert_close(fit, expected, atol=.002, rtol=.01)


def test_replacement_and_strict_restore():
    model = nn.Sequential(nn.PReLU(4), nn.PReLU(4))
    calibration = {str(i): dict(radius=4, coefficients=[[0., .6, .01]]) for i in range(2)}
    replace_shared(model, calibration)
    assert not any(isinstance(m, nn.PReLU) for m in model.modules())
    assert sum(m.coeffs.numel() for m in model if isinstance(m, SharedQuadratic)) == 6
    clone = nn.Sequential(*[SharedQuadratic(torch.zeros(4)) for _ in range(2)])
    clone.load_state_dict(model.state_dict(), strict=True)
    x = torch.randn(2, 4, 3, 3)
    torch.testing.assert_close(model.eval()(x), clone.eval()(x))


def test_full_model_has_exactly_75_coefficients():
    model = build_shared_iresnet50(dropout=0, fp16=False)
    modules = [m for m in model.modules() if isinstance(m, SharedQuadratic)]
    assert len(modules) == 25
    assert sum(m.coeffs.numel() for m in modules) == 75
    assert not any(isinstance(m, nn.PReLU) for m in model.modules())


def test_audit_detects_hidden_nonfinite_despite_finite_final_output():
    class Overflow(nn.Module):
        def forward(self, x):
            return x*float('inf')
    class Sanitize(nn.Module):
        def forward(self, x):
            return torch.nan_to_num(x)
    model = nn.Sequential(Overflow(), Sanitize())
    audit = FiniteAudit(model)
    assert torch.isfinite(model(torch.ones(2, 3))).all()
    assert audit.result()['nonfinite_values'] > 0
    assert audit.result()['boundaries']['0.output']['nonfinite_values'] == 6
    audit.close()

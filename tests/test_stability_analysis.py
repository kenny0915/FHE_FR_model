import torch
from torch import nn
from torch.utils.data import DataLoader

from stability_analysis.activations import HerPN, PolynomialActivation
from stability_analysis.workflow import AnalysisConfig, analyze, replace_activations


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(4, 4)
        self.act = nn.PReLU(4)
        self.linear2 = nn.Linear(4, 3)

    def forward(self, x):
        return self.linear2(self.act(self.linear1(x)))


def test_herpn_states_target_interval_and_degree():
    activation = HerPN(input_scale=6, prelu_slope=0.25)
    assert activation.interval == (-6.0, 6.0)
    assert activation.degree == 2
    assert "PReLU" in activation.target
    assert torch.isfinite(activation(torch.tensor([-6.0, 0.0, 6.0]))).all()


def test_custom_polynomial_uses_constant_first_coefficients():
    activation = PolynomialActivation([1, 2, 3], interval=(-2, 2))
    assert torch.allclose(activation(torch.tensor([2.0])), torch.tensor([17.0]))


def test_analysis_replaces_activation_and_collects_backward_proxy():
    torch.manual_seed(1)
    baseline = TinyModel()
    polynomial, replacements = replace_activations(baseline, input_scale=2)
    report = analyze(
        baseline, polynomial, DataLoader(torch.randn(6, 4), batch_size=2),
        replacements,
        config=AnalysisConfig(max_batches=2, interval=(-2, 2), backward=True),
    )
    assert list(replacements) == ["act"]
    assert report["batches_analyzed"] == 2
    assert report["activation_inputs"]["act"]["count"] == 16
    assert report["backward_proxy"]["gradient_tensors"] > 0
    assert report["backward_proxy"]["nonfinite_gradient_tensors"] == 0

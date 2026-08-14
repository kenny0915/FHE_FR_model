import torch
from torch import nn
from torch.utils.data import DataLoader

from stability_analysis.activations import (
    ChannelwisePolynomialActivation,
    HerPN,
    PolynomialActivation,
    make_herpn_for,
    make_uniform_quadratic_for,
)
from stability_analysis.layerwise import TensorStats, analyze_layerwise
from stability_analysis.reporting import render_markdown
from stability_analysis.workflow import AnalysisConfig, analyze, replace_activations


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(4, 4)
        self.act = nn.PReLU(4)
        self.linear2 = nn.Linear(4, 3)

    def forward(self, x):
        return self.linear2(self.act(self.linear1(x)))


class TinyTwoActivationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(4, 4)
        self.act1 = nn.PReLU(4)
        self.linear2 = nn.Linear(4, 4)
        self.act2 = nn.ReLU()
        self.linear3 = nn.Linear(4, 3)

    def forward(self, x):
        x = self.act1(self.linear1(x))
        x = self.act2(self.linear2(x))
        return self.linear3(x)


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


def test_channelwise_herpn_preserves_each_prelu_slope():
    prelu = nn.PReLU(3)
    with torch.no_grad():
        prelu.weight.copy_(torch.tensor([0.1, 0.25, 0.4]))
    activation = make_herpn_for(prelu, input_scale=5)

    assert isinstance(activation, ChannelwisePolynomialActivation)
    assert activation.coefficients.shape == (3, 3)
    assert activation.interval == (-5.0, 5.0)
    assert not torch.allclose(
        activation.coefficients[:, 0], activation.coefficients[:, 2])
    assert activation(torch.randn(2, 3, 2, 2)).shape == (2, 3, 2, 2)


def test_uniform_quadratic_interval_trades_center_for_tail_error():
    relu = nn.ReLU()
    narrow = make_uniform_quadratic_for(relu, input_scale=2)
    wide = make_uniform_quadratic_for(relu, input_scale=8)
    center = torch.tensor([0.0])
    outlier = torch.tensor([8.0])

    assert narrow(center).abs() < wide(center).abs()
    assert (narrow(outlier) - outlier).abs() > (
        wide(outlier) - outlier).abs()
    assert narrow.coefficients[2] > wide.coefficients[2]


def test_layerwise_study_replaces_each_activation_and_renders_report():
    torch.manual_seed(4)
    report = analyze_layerwise(
        TinyTwoActivationModel(),
        DataLoader(torch.randn(6, 4), batch_size=3),
        max_batches=2,
        interval=(-2, 2),
        interval_scales=(1, 2, 4),
        max_samples=128,
    )
    report["run"] = {
        "model": "tiny",
        "checkpoint": "trained.pt",
        "dataset": "synthetic",
        "dataset_kind": "synthetic",
    }

    assert report["activation_count"] == 2
    assert [row["name"] for row in report["layer_results"]] == [
        "act1", "act2"]
    assert report["layer_results"][0]["downstream_probe"] == "act2"
    assert report["layer_results"][1]["downstream_probe"] == "embedding"
    assert len(report["interval_summary"]) == 3
    assert len(report["rankings"]["least_embedding_effect"]) == 2
    assert "model_behavior" in report["all_replaced"]
    markdown = render_markdown(report)
    assert "Smoke-result warning" in markdown
    assert "`act1`" in markdown
    assert "Approximation interval sweep" in markdown


def test_tensor_stats_sample_represents_later_batches():
    stats = TensorStats(interval=(-2, 2), max_samples=4)
    stats.add(torch.zeros(4, 1))
    stats.add(torch.ones(4, 1))
    values, _ = stats.samples()

    assert values.numel() == 4
    assert torch.count_nonzero(values == 0) == 2
    assert torch.count_nonzero(values == 1) == 2

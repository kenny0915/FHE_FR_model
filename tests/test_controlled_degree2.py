import copy

import pytest
import torch
from torch.nn import functional as F

from controlled_degree2.calibrate import reference_ranges, weighted_quadratic_abs_fit
from controlled_degree2.augment import prepare_range_batch
from controlled_degree2.model import (
    DirectQuadratic,
    prelu_to_quadratic_coefficients,
    scale_intervals,
)


def test_uniform_degree2_fit_has_expected_closed_form():
    slope = torch.tensor([0.0, 0.25])
    coefficients = prelu_to_quadratic_coefficients(slope, [2.0, 4.0])

    expected_even = torch.tensor([[3.0 / 8.0, 15.0 / 32.0],
                                  [3.0 / 4.0, 15.0 / 64.0]])
    expected = torch.stack(
        ((1 - slope) / 2 * expected_even[:, 0],
         (1 + slope) / 2,
         (1 - slope) / 2 * expected_even[:, 1]),
        dim=1,
    )
    assert torch.allclose(coefficients, expected)


def test_training_alpha_zero_is_exact_prelu_and_collects_hinge():
    slopes = torch.tensor([0.1, 0.4])
    activation = DirectQuadratic(2, lam_fit=[2.0, 3.0], lam_reg=[1.0, 1.5], slope=slopes)
    activation.alpha = 0.0
    activation.train()
    inputs = torch.tensor([[[[-2.0]], [[2.0]]]], requires_grad=True)

    actual = activation(inputs)
    expected = F.prelu(inputs, slopes)

    assert torch.equal(actual, expected)
    assert activation.last_oor == pytest.approx(1.0)
    assert activation.last_penalty.item() > 0


def test_eval_is_unclipped_but_optional_diagnostic_clip_is_bounded():
    activation = DirectQuadratic(1, lam_fit=2.0, lam_reg=1.2, slope=0.25).eval()
    inputs = torch.tensor([[[[20.0]]]])
    deployable = activation(inputs)
    activation.clip_eval = True
    diagnostic = activation(inputs)

    assert deployable.item() > diagnostic.item() * 5


def test_interval_scaling_preserves_rescaled_polynomial_shape():
    activation = DirectQuadratic(2, lam_fit=[2.0, 4.0], lam_reg=[1.2, 2.4], slope=[0.1, 0.3])
    original = copy.deepcopy(activation).eval()
    calibration = {
        "act": {
            "lam_fit": [2.0, 4.0],
            "lam_reg": [1.2, 2.4],
            "even_coeffs": [[0.3, 0.4], [0.5, 0.2]],
        }
    }
    activation.name = "act"
    touched = scale_intervals(activation, calibration, {"act": 1.5})
    inputs = torch.randn(3, 2, 4, 4)

    assert touched == {"act": 1.5}
    assert torch.allclose(
        activation.eval()(inputs), 1.5 * original(inputs / 1.5), rtol=1e-6, atol=1e-6
    )
    assert calibration["act"]["lam_fit"] == pytest.approx([3.0, 6.0])


def test_histogram_weighted_fit_returns_a_finite_direct_quadratic():
    centers = torch.logspace(-3, 1, 512, dtype=torch.float64)
    widths = torch.diff(torch.cat((centers[:1] / 1.01, centers)))
    probabilities = torch.exp(-0.5 * (centers[None, :] / torch.tensor([[0.8], [1.4]])) ** 2)
    probabilities /= probabilities.sum(dim=1, keepdim=True)
    coefficients, errors = weighted_quadratic_abs_fit(
        probabilities, centers, widths, torch.tensor([3.0, 5.0]), fit_eps=0.05
    )

    assert coefficients.shape == (2, 2)
    assert errors.shape == (2,)
    assert torch.isfinite(coefficients).all()
    assert torch.isfinite(errors).all()
    assert bool((errors < 0.6).all())


def test_reference_uses_deployed_range_buffers_and_widening_metadata(tmp_path):
    path = tmp_path / "run10.pt"
    torch.save(
        {
            "format": "reference",
            "state_dict": {
                "prelu.lam_fit": torch.tensor([2.0, 3.0]),
                "prelu.lam_reg": torch.tensor([1.2, 1.8]),
            },
            "poly_calib": {
                "prelu": {
                    "lam_fit": [99.0, 99.0],
                    "lam_reg": [98.0, 98.0],
                    "lam_scale": 1.5,
                }
            },
        },
        path,
    )

    ranges, checkpoint_format = reference_ranges(path, ["prelu"], {"prelu": 2})
    lam_fit, lam_reg, scale = ranges["prelu"]

    assert checkpoint_format == "reference"
    assert lam_fit.tolist() == [2.0, 3.0]
    assert lam_reg.tolist() == pytest.approx([1.2, 1.8])
    assert scale == 1.5


def test_registered_controlled_backbone_has_25_direct_quadratics():
    from backbones import get_model

    model = get_model("r50_controlled_d2", dropout=0, fp16=False)
    activations = [module for module in model.modules() if isinstance(module, DirectQuadratic)]

    assert len(activations) == 25
    assert all(module.coeffs.shape[1] == 3 for module in activations)


def test_range_coverage_keeps_realistic_domain_and_masks_pathological_rows():
    torch.manual_seed(9)
    images = torch.rand(4, 3, 112, 112) * 2.0 - 1.0
    output, mask = prepare_range_batch(
        images,
        pathological_fraction=0.25,
        crop_probability=1.0,
        lowres_probability=1.0,
        photo_probability=1.0,
        stress_probability=1.0,
    )

    assert output.shape == images.shape
    assert torch.isfinite(output).all()
    assert float(output.min()) >= -1.0
    assert float(output.max()) <= 1.0
    assert mask.tolist() == [True, True, True, False]

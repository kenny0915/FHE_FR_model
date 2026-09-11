import numpy as np
import pytest
import torch

from controlled_degree2.compare_quadratic_checkpoints import (
    activation_data,
    calibration_buffer_consistency,
    compare_activation,
    degree2_coefficients,
    extract_state_dict,
)


def _activation(coeffs, lam_fit, lam_reg=None):
    channels = len(lam_fit)
    if lam_reg is None:
        lam_reg = lam_fit
    return {
        "prelu.coeffs": torch.tensor(coeffs, dtype=torch.float32),
        "prelu.lam_fit": torch.tensor(lam_fit, dtype=torch.float32),
        "prelu.lam_reg": torch.tensor(lam_reg, dtype=torch.float32),
        "prelu.slope": torch.full((channels,), 0.25),
    }


def test_extract_state_dict_accepts_wrapped_and_plain_states():
    state = _activation([[0.1, 0.6, 0.2]], [2.0])
    assert extract_state_dict(state) is state
    assert extract_state_dict({"state_dict_backbone": state, "degree": 2}) is state
    assert extract_state_dict({"state_dict": state, "degree": 2}) is state


def test_degree2_coefficients_strips_only_zero_cubic_slot():
    padded = torch.tensor([[0.1, 0.6, 0.2, 0.0], [0.2, 0.5, 0.3, 0.0]])
    result = degree2_coefficients(padded, "prelu.coeffs")
    assert result.shape == (2, 3)
    np.testing.assert_allclose(result, padded[:, :3].numpy())
    padded[0, 3] = 0.01
    with pytest.raises(ValueError, match="not degree 2"):
        degree2_coefficients(padded, "prelu.coeffs")


def test_scaled_polynomials_match_in_normalized_coordinates():
    # q_right(x) = r*q_left(x/r), lam_right=r*lam_left.  Raw coefficients
    # differ, but the normalized curves must be identical.
    ratio = 0.4
    left_state = _activation([[0.2, 0.6, 0.3]], [2.0], [1.2])
    right_state = _activation(
        [[ratio * 0.2, 0.6, 0.3 / ratio]],
        [ratio * 2.0],
        [0.4 * ratio * 2.0],
    )
    left = activation_data(left_state)["prelu"]
    right = activation_data(right_state)["prelu"]
    result = compare_activation(left, right)
    assert result["right_over_left_lam_fit"]["median"] == pytest.approx(ratio)
    assert result["normalized_curve_rmse"]["max"] < 1e-7
    assert result["raw_curve_rmse_on_overlap"]["median"] > 0


def test_calibration_consistency_catches_stale_metadata():
    activations = activation_data(_activation([[0.1, 0.6, 0.2]], [2.0], [1.2]))
    metadata = {"prelu": {"lam_fit": [2.0], "lam_reg": [1.9]}}
    result = calibration_buffer_consistency(metadata, activations)
    assert result["lam_fit"]["mismatched_sites"] == []
    assert result["lam_reg"]["mismatched_sites"] == ["prelu"]

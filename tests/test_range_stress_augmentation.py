import pytest
import torch

from utils.utils_range_augmentation import RangeStressAugmentation


def test_range_stress_augmentation_is_finite_bounded_and_nontrivial():
    torch.manual_seed(17)
    augmentation = RangeStressAugmentation({
        "enabled": True,
        "probability": 1.0,
        "contrast": (1.8, 1.8),
        "gain": (1.35, 1.35),
        "bias": (-0.1, -0.1),
        "gamma": (0.7, 0.7),
        "noise_probability": 1.0,
        "noise_std": 0.04,
    })
    image = torch.linspace(0.0, 1.0, 3 * 16 * 16).reshape(3, 16, 16)

    augmented = augmentation(image)

    assert augmented.shape == image.shape
    assert augmented.dtype == image.dtype
    assert torch.isfinite(augmented).all()
    assert float(augmented.min()) >= 0.0
    assert float(augmented.max()) <= 1.0
    assert not torch.equal(augmented, image)


def test_disabled_range_stress_augmentation_is_exact_identity():
    image = torch.rand(3, 8, 8)
    augmentation = RangeStressAugmentation({
        "enabled": False,
        "probability": 1.0,
    })

    assert augmentation(image) is image


@pytest.mark.parametrize("field,value", [
    ("probability", 1.1),
    ("noise_probability", -0.1),
    ("contrast", (0.0, 1.0)),
    ("gamma", (1.0, 0.5)),
    ("noise_std", -0.01),
])
def test_invalid_range_stress_config_is_rejected(field, value):
    config = {"enabled": True, field: value}
    with pytest.raises(ValueError):
        RangeStressAugmentation(config)

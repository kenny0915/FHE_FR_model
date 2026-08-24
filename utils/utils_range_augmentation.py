"""Bounded training augmentation for polynomial activation range coverage."""

import torch


def range_augmentation_value(config, name, default):
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


def range_augmentation_enabled(config):
    return bool(range_augmentation_value(config, "enabled", False))


class RangeStressAugmentation:
    """Generate bounded exposure, contrast, gamma, and texture tails.

    This transform runs on a ``[0, 1]`` image tensor before normalization.
    It is a plaintext training augmentation and is not part of FHE inference.
    """

    def __init__(self, config=None):
        self.enabled = range_augmentation_enabled(config)
        self.probability = float(
            range_augmentation_value(config, "probability", 0.0))
        self.contrast = tuple(
            float(value) for value in
            range_augmentation_value(config, "contrast", (1.0, 1.0)))
        self.gain = tuple(
            float(value) for value in
            range_augmentation_value(config, "gain", (1.0, 1.0)))
        self.bias = tuple(
            float(value) for value in
            range_augmentation_value(config, "bias", (0.0, 0.0)))
        self.gamma = tuple(
            float(value) for value in
            range_augmentation_value(config, "gamma", (1.0, 1.0)))
        self.noise_probability = float(
            range_augmentation_value(config, "noise_probability", 0.0))
        self.noise_std = float(
            range_augmentation_value(config, "noise_std", 0.0))

        if not 0.0 <= self.probability <= 1.0:
            raise ValueError("range augmentation probability must be in [0, 1]")
        if not 0.0 <= self.noise_probability <= 1.0:
            raise ValueError(
                "range augmentation noise_probability must be in [0, 1]")
        for name, bounds in (
                ("contrast", self.contrast),
                ("gain", self.gain),
                ("bias", self.bias),
                ("gamma", self.gamma)):
            if len(bounds) != 2 or bounds[0] > bounds[1]:
                raise ValueError(
                    f"range augmentation {name} must be ordered bounds")
        if min(self.contrast) <= 0.0 or min(self.gain) <= 0.0:
            raise ValueError("range augmentation contrast/gain must be positive")
        if min(self.gamma) <= 0.0:
            raise ValueError("range augmentation gamma must be positive")
        if self.noise_std < 0.0:
            raise ValueError("range augmentation noise_std must be non-negative")

    @staticmethod
    def _uniform(bounds, image):
        low, high = bounds
        if low == high:
            return low
        return float(torch.empty((), device=image.device).uniform_(low, high))

    def __call__(self, image):
        if (not self.enabled or self.probability == 0.0
                or float(torch.rand(())) >= self.probability):
            return image
        if not torch.is_tensor(image) or image.ndim != 3:
            raise ValueError("range augmentation expects a CHW tensor")

        output = image.float()
        spatial_mean = output.mean(dim=(-2, -1), keepdim=True)
        contrast = self._uniform(self.contrast, output)
        gain = self._uniform(self.gain, output)
        bias = self._uniform(self.bias, output)
        gamma = self._uniform(self.gamma, output)
        output = (output - spatial_mean) * contrast + spatial_mean
        output = output * gain + bias
        output = output.clamp(0.0, 1.0).pow(gamma)
        if (self.noise_std > 0.0 and self.noise_probability > 0.0
                and float(torch.rand(())) < self.noise_probability):
            noise_scale = self.noise_std * float(torch.rand(()))
            output = output + noise_scale * torch.randn_like(output)
        return output.clamp_(0.0, 1.0).to(dtype=image.dtype)

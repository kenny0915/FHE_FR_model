"""GPU-side input coverage used in run10's stabilising fine-tune."""

import math

import torch
from torch.nn import functional as F
from torchvision.transforms import functional as TVF


def _uniform(n, low, high, device):
    return torch.rand(n, device=device) * (high - low) + low


def _lowres(images):
    output = torch.empty_like(images)
    for index in range(images.shape[0]):
        side = int(torch.randint(14, 48, (1,)))
        small = F.interpolate(
            images[index : index + 1], size=(side, side), mode="bilinear", align_corners=False
        )
        output[index] = F.interpolate(
            small, size=images.shape[-2:], mode="bilinear", align_corners=False
        )[0]
    return output


def _blur(images):
    kernel_size = 9
    sigma = _uniform(images.shape[0], 1.0, 3.5, images.device)
    axis = torch.arange(kernel_size, device=images.device).float() - 4
    gaussian = torch.exp(-(axis[None, :] ** 2) / (2 * sigma[:, None] ** 2))
    gaussian /= gaussian.sum(dim=1, keepdim=True)
    kernels = gaussian[:, :, None] * gaussian[:, None, :]
    output = torch.empty_like(images)
    for index in range(images.shape[0]):
        weight = kernels[index][None, None].expand(3, 1, kernel_size, kernel_size)
        output[index] = F.conv2d(
            images[index : index + 1], weight, padding=4, groups=3
        )[0]
    return output


def _photo(images):
    output = torch.empty_like(images)
    unit = (images + 1.0) / 2.0
    for index in range(images.shape[0]):
        factors = _uniform(3, 0.5, 1.5, images.device)
        order = torch.randperm(3)
        sample = unit[index]
        for operation in order:
            if operation == 0:
                sample = TVF.adjust_brightness(sample, float(factors[0]))
            elif operation == 1:
                sample = TVF.adjust_contrast(sample, float(factors[1]))
            else:
                sample = TVF.adjust_saturation(sample, float(factors[2]))
        output[index] = sample * 2.0 - 1.0
    return output


def _dark(images):
    scale = _uniform(images.shape[0], 0.1, 0.5, images.device).view(-1, 1, 1, 1)
    return (((images + 1.0) * 0.5 * scale) * 2.0 - 1.0).clamp(-1.0, 1.0)


def _bright(images):
    scale = _uniform(images.shape[0], 1.5, 3.0, images.device).view(-1, 1, 1, 1)
    return (((images + 1.0) * 0.5 * scale) * 2.0 - 1.0).clamp(-1.0, 1.0)


def _contrast(images):
    mean = images.mean(dim=(1, 2, 3), keepdim=True)
    scale = _uniform(images.shape[0], 2.0, 4.0, images.device).view(-1, 1, 1, 1)
    return ((images - mean) * scale + mean).clamp(-1.0, 1.0)


def _lowcontrast(images):
    mean = images.mean(dim=(1, 2, 3), keepdim=True)
    scale = _uniform(images.shape[0], 0.1, 0.4, images.device).view(-1, 1, 1, 1)
    return (images - mean) * scale + mean


def _noise(images):
    sigma = _uniform(images.shape[0], 0.05, 0.25, images.device).view(-1, 1, 1, 1)
    return (images + torch.randn_like(images) * sigma).clamp(-1.0, 1.0)


def _erase(images):
    output = images.clone()
    height, width = images.shape[-2:]
    for index in range(images.shape[0]):
        h = int(torch.randint(30, min(80, height), (1,)))
        w = int(torch.randint(30, min(80, width), (1,)))
        top = int(torch.randint(0, height - h + 1, (1,)))
        left = int(torch.randint(0, width - w + 1, (1,)))
        output[index, :, top : top + h, left : left + w] = float(torch.rand(1) * 2 - 1)
    return output


def _jpegish(images):
    return F.interpolate(F.avg_pool2d(images, 4), size=images.shape[-2:], mode="nearest")


def _combo(images):
    return _noise(_dark(_lowres(images)))


# This is the run10 stress family; the separate AdaFace-style lowres/photo
# controls below are sampled independently, as they were in run10.
DEGRADATIONS = (
    _lowres,
    _blur,
    _dark,
    _bright,
    _contrast,
    _lowcontrast,
    _noise,
    _erase,
    _jpegish,
    _combo,
)


@torch.no_grad()
def random_degrade(images, probability: float):
    if probability <= 0:
        return images
    selected = (torch.rand(images.shape[0], device=images.device) < probability).nonzero().flatten()
    if not selected.numel():
        return images
    which = torch.randint(len(DEGRADATIONS), (selected.numel(),), device=images.device)
    output = images.clone()
    for index, degradation in enumerate(DEGRADATIONS):
        rows = selected[which == index]
        if rows.numel():
            output[rows] = degradation(images[rows])
    return output


@torch.no_grad()
def _crop_with_black_padding(images):
    """AdaFace crop augmentation in normalized RGB tensor space."""
    output = torch.full_like(images, -1.0)
    height, width = images.shape[-2:]
    area = height * width
    for index in range(images.shape[0]):
        crop_h = crop_w = None
        for _ in range(10):
            target_area = float(_uniform(1, 0.2, 1.0, images.device)) * area
            ratio = math.exp(
                float(
                    _uniform(
                        1,
                        math.log(0.75),
                        math.log(4.0 / 3.0),
                        images.device,
                    )
                )
            )
            candidate_w = round(math.sqrt(target_area * ratio))
            candidate_h = round(math.sqrt(target_area / ratio))
            if 0 < candidate_h <= height and 0 < candidate_w <= width:
                crop_h, crop_w = candidate_h, candidate_w
                break
        if crop_h is None:
            crop_h, crop_w = height, width
        top = int(torch.randint(0, height - crop_h + 1, (1,)))
        left = int(torch.randint(0, width - crop_w + 1, (1,)))
        output[index, :, top : top + crop_h, left : left + crop_w] = images[
            index, :, top : top + crop_h, left : left + crop_w
        ]
    return output


def run10_coverage_augmentation(
    images,
    crop_probability=0.1,
    lowres_probability=0.2,
    photo_probability=0.2,
    stress_probability=0.4,
):
    """Apply the three independently sampled run10 augmentation controls."""
    output = images
    crop_rows = torch.rand(images.shape[0], device=images.device) < crop_probability
    if crop_rows.any():
        output = output.clone()
        output[crop_rows] = _crop_with_black_padding(output[crop_rows])
    lowres_rows = torch.rand(images.shape[0], device=images.device) < lowres_probability
    if lowres_rows.any():
        output = output.clone()
        output[lowres_rows] = _lowres(output[lowres_rows])
    photo_rows = torch.rand(images.shape[0], device=images.device) < photo_probability
    if photo_rows.any():
        output = output.clone()
        output[photo_rows] = _photo(output[photo_rows])
    return random_degrade(output, stress_probability)


@torch.no_grad()
def random_pathological(count: int, device, height=112, width=112):
    output = torch.empty(count, 3, height, width, device=device)
    yy, xx = torch.meshgrid(
        torch.arange(height, device=device),
        torch.arange(width, device=device),
        indexing="ij",
    )
    for index in range(count):
        kind = int(torch.randint(5, (1,)))
        amplitude = float(_uniform(1, 0.5, 1.0, device))
        if kind == 0:
            image = torch.where(
                torch.rand(3, height, width, device=device) < 0.5,
                -amplitude,
                amplitude,
            )
        elif kind == 1:
            cell = int(torch.randint(2, 12, (1,)))
            image = (((yy // cell + xx // cell) % 2) * 2.0 - 1.0)[None].expand(
                3, height, width
            ) * amplitude
        elif kind == 2:
            cell = int(torch.randint(1, 8, (1,)))
            base = xx if torch.rand(1) < 0.5 else yy
            image = (((base // cell) % 2) * 2.0 - 1.0)[None].expand(
                3, height, width
            ) * amplitude
        elif kind == 3:
            image = (torch.rand(3, height, width, device=device) * 2.0 - 1.0) * amplitude
        else:
            image = torch.full(
                (3, height, width), float(_uniform(1, -1, 1, device)), device=device
            )
        output[index] = image
    return output


def prepare_range_batch(images, pathological_fraction=0.05, **augmentation):
    images = run10_coverage_augmentation(images, **augmentation)
    mask = torch.ones(images.shape[0], dtype=torch.bool, device=images.device)
    if pathological_fraction > 0:
        count = max(1, int(round(images.shape[0] * pathological_fraction)))
        images = images.clone()
        images[-count:] = random_pathological(
            count, images.device, images.shape[-2], images.shape[-1]
        ).to(images.dtype)
        mask[-count:] = False
    return images, mask

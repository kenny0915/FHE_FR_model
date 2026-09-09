"""Deterministic non-IJB datasets for deployment-tail mining and replay."""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor


@dataclass(frozen=True)
class WiderFaceRecord:
    relative_path: str
    x: int
    y: int
    width: int
    height: int
    context_scale: float
    stress_variant: str


IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def parse_context_scales(spec) -> tuple[float, ...]:
    if isinstance(spec, str):
        values = tuple(
            float(value)
            for value in re.split(r"[,+;\s]+", spec)
            if value.strip()
        )
    else:
        values = tuple(float(value) for value in spec)
    if not values or any(value <= 0.0 for value in values):
        raise ValueError("WIDER context scales must be positive")
    return values


WIDER_STRESS_VARIANTS = (
    "base",
    "lowres14",
    "lowres7",
    "dark",
    "bright",
    "contrast",
    "lowcontrast",
    "gamma035",
    "gamma25",
    "shift_left",
    "shift_right",
    "shift_up",
    "shift_down",
    "shift16_left",
    "shift16_right",
    "shift16_up",
    "shift16_down",
    "zoomout75",
    "jpeg4",
    "occlude",
)


def parse_stress_variants(spec) -> tuple[str, ...]:
    if isinstance(spec, str):
        values = tuple(
            value for value in re.split(r"[,+;\s]+", spec) if value
        )
    else:
        values = tuple(str(value) for value in spec)
    if not values:
        raise ValueError("at least one deployment stress variant is required")
    unknown = sorted(set(values) - set(WIDER_STRESS_VARIANTS))
    if unknown:
        raise ValueError(f"unknown deployment stress variants: {unknown}")
    return values


def parse_wider_annotations(
    path,
    context_scales=(1.0, 1.5),
    stress_variants=("base",),
):
    """Parse valid WIDER train boxes in official deterministic file order."""
    scales = parse_context_scales(context_scales)
    variants = parse_stress_variants(stress_variants)
    records = []
    with open(path, encoding="utf-8") as handle:
        lines = iter(handle)
        while True:
            try:
                relative_path = next(lines).strip()
            except StopIteration:
                break
            if not relative_path:
                continue
            try:
                count = int(next(lines).strip())
            except StopIteration as error:
                raise ValueError("truncated WIDER annotation count") from error
            if count == 0:
                # The official WIDER train file stores one all-zero dummy box
                # after each zero-face count.  It is not part of the next
                # record and must be consumed to keep the stream aligned.
                try:
                    dummy = [int(value) for value in next(lines).split()]
                except StopIteration as error:
                    raise ValueError("truncated WIDER zero-face record") from error
                if not dummy or any(dummy):
                    raise ValueError(
                        f"expected all-zero WIDER dummy box, got {dummy}"
                    )
                continue
            for _ in range(count):
                try:
                    fields = [int(value) for value in next(lines).split()]
                except StopIteration as error:
                    raise ValueError("truncated WIDER bounding-box records") from error
                if len(fields) < 8:
                    raise ValueError(f"malformed WIDER bounding box: {fields}")
                x, y, width, height = fields[:4]
                invalid = fields[7]
                if invalid or width <= 0 or height <= 0:
                    continue
                records.extend(
                    WiderFaceRecord(
                        relative_path,
                        x,
                        y,
                        width,
                        height,
                        scale,
                        variant,
                    )
                    for scale in scales
                    for variant in variants
                )
    if not records:
        raise ValueError(f"no valid WIDER faces parsed from {path}")
    return tuple(records)


class WiderFaceCropDataset(Dataset):
    """Square WIDER face-box crops resized to the recognition input size."""

    def __init__(
        self,
        image_root,
        annotations,
        context_scales=(1.0, 1.5),
        stress_variants=("base",),
        image_size=112,
    ):
        self.image_root = os.path.abspath(image_root)
        self.annotations = os.path.abspath(annotations)
        self.records = parse_wider_annotations(
            annotations, context_scales, stress_variants
        )
        self.image_size = int(image_size)
        if self.image_size <= 0:
            raise ValueError("image size must be positive")

    def __len__(self):
        return len(self.records)

    def _read(self, index):
        record = self.records[index]
        path = os.path.join(self.image_root, record.relative_path)
        with Image.open(path) as source:
            image = source.convert("RGB")
            center_x = record.x + record.width / 2.0
            center_y = record.y + record.height / 2.0
            side = max(record.width, record.height) * record.context_scale
            left = round(center_x - side / 2.0)
            top = round(center_y - side / 2.0)
            right = round(center_x + side / 2.0)
            bottom = round(center_y + side / 2.0)
            if right <= left:
                right = left + 1
            if bottom <= top:
                bottom = top + 1
            # PIL pads out-of-image regions with black, matching the border
            # behavior of IJB landmark warping without reading any IJB data.
            image = image.crop((left, top, right, bottom)).resize(
                (self.image_size, self.image_size), Image.Resampling.BILINEAR
            )
            tensor = pil_to_tensor(image).float() / 127.5 - 1.0
            tensor = apply_image_stress(tensor, record.stress_variant)
        return tensor, torch.tensor(0, dtype=torch.long)

    def __getitem__(self, index):
        return self._read(index)

    def get_oriented(self, index, orientation):
        image, label = self._read(index)
        if int(orientation):
            image = torch.flip(image, dims=(-1,))
        return image, label


def apply_image_stress(image, variant):
    """Apply one fixed stress transform in normalized RGB tensor space."""
    if variant == "base":
        return image
    if variant in ("lowres14", "lowres7"):
        side = 14 if variant == "lowres14" else 7
        small = torch.nn.functional.interpolate(
            image[None], size=(side, side), mode="bilinear", align_corners=False
        )
        return torch.nn.functional.interpolate(
            small, size=image.shape[-2:], mode="bilinear", align_corners=False
        )[0]
    if variant in ("dark", "bright"):
        factor = 0.15 if variant == "dark" else 2.5
        unit = ((image + 1.0) * 0.5 * factor).clamp(0.0, 1.0)
        return unit * 2.0 - 1.0
    if variant in ("contrast", "lowcontrast"):
        factor = 4.0 if variant == "contrast" else 0.15
        mean = image.mean()
        return ((image - mean) * factor + mean).clamp(-1.0, 1.0)
    if variant in ("gamma035", "gamma25"):
        exponent = 0.35 if variant == "gamma035" else 2.5
        unit = ((image + 1.0) * 0.5).clamp(0.0, 1.0)
        return unit.pow(exponent) * 2.0 - 1.0
    if variant.startswith("shift_") or variant.startswith("shift16_"):
        output = torch.full_like(image, -1.0)
        shift = 16 if variant.startswith("shift16_") else 8
        direction = variant.split("_", 1)[1]
        if direction == "left":
            output[:, :, :-shift] = image[:, :, shift:]
        elif direction == "right":
            output[:, :, shift:] = image[:, :, :-shift]
        elif direction == "up":
            output[:, :-shift, :] = image[:, shift:, :]
        elif direction == "down":
            output[:, shift:, :] = image[:, :-shift, :]
        else:
            raise ValueError(f"unknown deployment stress variant {variant!r}")
        return output
    if variant == "zoomout75":
        height, width = image.shape[-2:]
        resized_height = max(1, round(height * 0.75))
        resized_width = max(1, round(width * 0.75))
        resized = torch.nn.functional.interpolate(
            image[None],
            size=(resized_height, resized_width),
            mode="bilinear",
            align_corners=False,
        )[0]
        output = torch.full_like(image, -1.0)
        top = (height - resized_height) // 2
        left = (width - resized_width) // 2
        output[:, top:top + resized_height, left:left + resized_width] = resized
        return output
    if variant == "jpeg4":
        small = torch.nn.functional.avg_pool2d(image[None], 4)
        return torch.nn.functional.interpolate(
            small, size=image.shape[-2:], mode="nearest"
        )[0]
    if variant == "occlude":
        output = image.clone()
        output[:, 36:76, 28:84] = -1.0
        return output
    raise ValueError(f"unknown deployment stress variant {variant!r}")


# Backward-compatible public name for existing WIDER callers/tests.
apply_wider_stress = apply_image_stress


class AlignedImageDataset(Dataset):
    """Deterministic recursive loader for already-aligned non-IJB face frames.

    YouTube Faces' aligned archive is arranged in nested identity/video
    directories. Identity labels are intentionally discarded: these images
    are used only as unlabeled numerical deployment-tail probes and replay.
    """

    def __init__(
        self,
        image_root,
        stress_variants=("base",),
        image_size=112,
        image_suffixes=IMAGE_SUFFIXES,
    ):
        self.image_root = os.path.abspath(image_root)
        self.image_size = int(image_size)
        if self.image_size <= 0:
            raise ValueError("image size must be positive")
        self.stress_variants = parse_stress_variants(stress_variants)
        suffixes = tuple(str(value).lower() for value in image_suffixes)
        if not suffixes:
            raise ValueError("at least one aligned-image suffix is required")
        relative_paths = []
        for directory, directory_names, file_names in os.walk(self.image_root):
            directory_names.sort()
            for file_name in sorted(file_names):
                if file_name.lower().endswith(suffixes):
                    relative_paths.append(os.path.relpath(
                        os.path.join(directory, file_name), self.image_root
                    ))
        self.relative_paths = tuple(relative_paths)
        if not self.relative_paths:
            raise ValueError(
                f"no aligned face images found recursively under {self.image_root}"
            )

    def __len__(self):
        return len(self.relative_paths) * len(self.stress_variants)

    @property
    def index_digest(self):
        """Fingerprint the exact source-index ordering used by manifests."""
        digest = hashlib.sha256()
        for relative_path in self.relative_paths:
            digest.update(relative_path.encode("utf-8", errors="surrogateescape"))
            digest.update(b"\0")
        for variant in self.stress_variants:
            digest.update(variant.encode("ascii"))
            digest.update(b"\0")
        return digest.hexdigest()

    def _read(self, index):
        path_index, variant_index = divmod(index, len(self.stress_variants))
        relative_path = self.relative_paths[path_index]
        variant = self.stress_variants[variant_index]
        path = os.path.join(self.image_root, relative_path)
        with Image.open(path) as source:
            image = source.convert("RGB")
            if image.size != (self.image_size, self.image_size):
                image = image.resize(
                    (self.image_size, self.image_size), Image.Resampling.BILINEAR
                )
            tensor = pil_to_tensor(image).float() / 127.5 - 1.0
        return apply_image_stress(tensor, variant), torch.tensor(0, dtype=torch.long)

    def __getitem__(self, index):
        return self._read(index)

    def get_oriented(self, index, orientation):
        image, label = self._read(index)
        if int(orientation):
            image = torch.flip(image, dims=(-1,))
        return image, label


def build_deployment_dataset(
    dataset_type,
    root,
    *,
    local_rank=0,
    annotations=None,
    wider_context_scales=(1.0, 1.5),
    wider_stress_variants=("base",),
    aligned_stress_variants=("base",),
):
    if dataset_type == "ms1mv3":
        from dataset import MXFaceDataset

        return MXFaceDataset(root, local_rank=local_rank)
    if dataset_type == "wider":
        if not annotations:
            raise ValueError("WIDER deployment data requires --annotations")
        return WiderFaceCropDataset(
            root,
            annotations,
            context_scales=wider_context_scales,
            stress_variants=wider_stress_variants,
        )
    if dataset_type == "ytf":
        return AlignedImageDataset(
            root,
            stress_variants=aligned_stress_variants,
        )
    raise ValueError(f"unknown deployment dataset type {dataset_type!r}")

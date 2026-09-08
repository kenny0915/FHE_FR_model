"""Deterministic non-IJB datasets for deployment-tail mining and replay."""

from __future__ import annotations

import os
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


def parse_context_scales(spec) -> tuple[float, ...]:
    if isinstance(spec, str):
        values = tuple(float(value) for value in spec.split(",") if value.strip())
    else:
        values = tuple(float(value) for value in spec)
    if not values or any(value <= 0.0 for value in values):
        raise ValueError("WIDER context scales must be positive")
    return values


def parse_wider_annotations(path, context_scales=(1.0, 1.5)):
    """Parse valid WIDER train boxes in official deterministic file order."""
    scales = parse_context_scales(context_scales)
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
                    WiderFaceRecord(relative_path, x, y, width, height, scale)
                    for scale in scales
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
        image_size=112,
    ):
        self.image_root = os.path.abspath(image_root)
        self.annotations = os.path.abspath(annotations)
        self.records = parse_wider_annotations(annotations, context_scales)
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
        return tensor, torch.tensor(0, dtype=torch.long)

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
        )
    raise ValueError(f"unknown deployment dataset type {dataset_type!r}")

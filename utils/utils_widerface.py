"""Deterministic WIDER FACE crops for label-free numerical calibration."""

import hashlib
import os
from functools import lru_cache

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def parse_wider_face_annotations(path):
    """Parse the official bbox text, including its four zero-face sentinels."""
    with open(os.fspath(path), encoding="utf-8") as handle:
        lines = tuple(line.strip() for line in handle if line.strip())
    records = []
    position = 0
    while position < len(lines):
        relative_path = lines[position]
        position += 1
        if not relative_path.lower().endswith(".jpg"):
            raise ValueError(
                f"Expected a WIDER image path at line {position}: "
                f"{relative_path!r}")
        try:
            declared = int(lines[position])
        except (IndexError, ValueError) as error:
            raise ValueError(
                f"Invalid WIDER face count after {relative_path}") from error
        position += 1
        physical_boxes = []
        while (position < len(lines)
               and not lines[position].lower().endswith(".jpg")):
            fields = tuple(int(value) for value in lines[position].split())
            if len(fields) != 10:
                raise ValueError(
                    f"Expected 10 WIDER bbox fields at line {position + 1}")
            physical_boxes.append(fields)
            position += 1
        # The official file contains one all-zero sentinel line for each of
        # four images declared to contain zero faces.  Only the declared rows
        # are annotations.
        if len(physical_boxes) < declared:
            raise ValueError(
                f"WIDER record {relative_path} declares {declared} faces but "
                f"contains {len(physical_boxes)} bbox rows")
        records.append((relative_path, tuple(physical_boxes[:declared])))
    if not records:
        raise ValueError(f"Empty WIDER annotation file: {path}")
    return tuple(records)


def wider_image_fold(relative_path, modulo=10):
    """Return a stable image-level fold without depending on Python hashing."""
    modulo = int(modulo)
    if modulo <= 1:
        raise ValueError("WIDER split modulo must be greater than one")
    digest = hashlib.sha256(relative_path.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % modulo


def crop_wider_face(image, box, crop_scale=1.35, image_size=112):
    """Make a square, mean-padded crop from an official WIDER bounding box."""
    if image is None or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("WIDER crop expects a BGR image with three channels")
    x, y, width, height = (float(value) for value in box[:4])
    crop_scale = float(crop_scale)
    image_size = int(image_size)
    if width <= 0 or height <= 0 or crop_scale <= 0 or image_size <= 0:
        raise ValueError("WIDER box, crop scale, and output size must be positive")
    center_x = x + 0.5 * width
    center_y = y + 0.5 * height
    side = crop_scale * max(width, height)
    resize = image_size / side
    matrix = np.array([
        [resize, 0.0, 0.5 * image_size - resize * center_x],
        [0.0, resize, 0.5 * image_size - resize * center_y],
    ], dtype=np.float32)
    crop = cv2.warpAffine(
        image, matrix, (image_size, image_size),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
        borderValue=(127.5, 127.5, 127.5))
    return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)


class WIDERFaceDataset(Dataset):
    """Return deterministic bbox crops split at the original-image level.

    The official WIDER attributes are kept in ``samples`` for audit and
    stratified analysis, but no identity or attribute label enters training.
    """

    def __init__(
            self, image_root, annotation_path, split="calibration", *,
            validation_modulo=10, validation_fold=0, min_face_size=20,
            crop_scale=1.35, image_size=112):
        self.image_root = os.fspath(image_root)
        self.annotation_path = os.fspath(annotation_path)
        self.split = str(split).lower()
        if self.split not in ("calibration", "validation", "all"):
            raise ValueError(
                "WIDER split must be calibration, validation, or all")
        self.validation_modulo = int(validation_modulo)
        self.validation_fold = int(validation_fold)
        if not 0 <= self.validation_fold < self.validation_modulo:
            raise ValueError("WIDER validation fold is outside the modulo")
        self.min_face_size = int(min_face_size)
        self.crop_scale = float(crop_scale)
        self.image_size = int(image_size)
        if self.min_face_size <= 0:
            raise ValueError("WIDER minimum face size must be positive")

        samples = []
        image_paths = set()
        for relative_path, boxes in parse_wider_face_annotations(
                self.annotation_path):
            fold = wider_image_fold(relative_path, self.validation_modulo)
            if self.split == "calibration" and fold == self.validation_fold:
                continue
            if self.split == "validation" and fold != self.validation_fold:
                continue
            image_path = os.path.join(self.image_root, relative_path)
            if not os.path.isfile(image_path):
                raise FileNotFoundError(image_path)
            for box in boxes:
                x, y, width, height, blur, expression, illumination, invalid, occlusion, pose = box
                if invalid or width <= 0 or height <= 0:
                    continue
                if min(width, height) < self.min_face_size:
                    continue
                samples.append({
                    "relative_path": relative_path,
                    "box": (x, y, width, height),
                    "blur": blur,
                    "expression": expression,
                    "illumination": illumination,
                    "occlusion": occlusion,
                    "pose": pose,
                })
                image_paths.add(relative_path)
        if not samples:
            raise ValueError(
                f"WIDER {self.split} split contains no eligible faces")
        self.samples = tuple(samples)
        self.image_count = len(image_paths)

    def __len__(self):
        return len(self.samples)

    @lru_cache(maxsize=8)
    def _read_image(self, relative_path):
        path = os.path.join(self.image_root, relative_path)
        image = cv2.imread(path)
        if image is None:
            raise FileNotFoundError(path)
        return image

    def _canonical(self, index):
        sample = self.samples[int(index)]
        image = self._read_image(sample["relative_path"])
        crop = crop_wider_face(
            image, sample["box"], self.crop_scale, self.image_size)
        # ``torch.from_numpy`` is unavailable when an older PyTorch build is
        # loaded beside NumPy 2.x.  A writable byte buffer avoids that ABI
        # bridge and is equally deterministic for uint8 image data.
        buffer = bytearray(np.ascontiguousarray(crop).tobytes())
        tensor = torch.frombuffer(buffer, dtype=torch.uint8).reshape(
            self.image_size, self.image_size, 3).permute(2, 0, 1).float()
        return tensor / 127.5 - 1.0

    def __getitem__(self, index):
        index = int(index)
        return self._canonical(index), index

    def get_oriented(self, index, orientation):
        index = int(index)
        image = self._canonical(index)
        if int(orientation):
            image = torch.flip(image, dims=(-1,))
        return image, index

    def get_pair(self, index):
        image = self._canonical(int(index))
        return torch.stack((image, torch.flip(image, dims=(-1,))))

"""Deterministic IJB aligned-image datasets and replay manifests."""

import csv
import json
import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


_ALIGNMENT_TEMPLATE = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)


def normalize_orientation(value):
    if value in (0, "0", "original"):
        return 0
    if value in (1, "1", "flip"):
        return 1
    raise ValueError(f"Invalid IJB orientation: {value!r}")


def load_ijbc_replay_orientations(paths, activation_topk=0):
    """Load unique ``(source_index, orientation)`` rows from CSV/JSON.

    CSV inputs are the non-finite manifests written by ``eval_ijbc.py``.
    JSON inputs may additionally contain per-activation finite tails written by
    ``mine_ijbc_herpn_tails.py``.  Exact output failures are always included.
    """
    if isinstance(paths, (str, os.PathLike)):
        paths = (paths,)
    activation_topk = int(activation_topk)
    if activation_topk < 0:
        raise ValueError("activation_topk must be non-negative")
    rows = []
    for path in paths:
        path = os.fspath(path)
        if path.lower().endswith(".csv"):
            with open(path, newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    rows.append((
                        int(row["source_index"]),
                        normalize_orientation(row["orientation"]),
                    ))
            continue
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise ValueError(f"IJB replay manifest must be an object: {path}")
        for row in payload.get("output_nonfinite", ()):
            rows.append((
                int(row["source_index"]),
                normalize_orientation(row["orientation"]),
            ))
        for row in payload.get("range_violations", ()):
            rows.append((
                int(row["source_index"]),
                normalize_orientation(row["orientation"]),
            ))
        if activation_topk:
            for activation in payload.get("activations", {}).values():
                for row in activation.get("tail", ())[:activation_topk]:
                    rows.append((
                        int(row["source_index"]),
                        normalize_orientation(row["orientation"]),
                    ))
    unique = tuple(dict.fromkeys(rows))
    if not unique:
        raise ValueError("IJB replay manifests contain no orientations")
    if any(index < 0 for index, _ in unique):
        raise ValueError("IJB source indices must be non-negative")
    return unique


def _read_metadata(root, target):
    path = os.path.join(
        os.fspath(root), "meta", f"{str(target).lower()}_name_5pts_score.txt")
    with open(path, encoding="utf-8") as handle:
        lines = tuple(line.rstrip("\n") for line in handle)
    if not lines:
        raise ValueError(f"Empty IJB metadata: {path}")
    return lines


def align_ijbc_pair(image, landmarks):
    """Match the original ``eval_ijbc.py`` alignment and normalization."""
    # Keep the optional IJB/scikit-image dependency out of MS1Mv3 and WIDER
    # calibration processes.  Some deployment environments intentionally
    # install only the training data stack.
    from skimage import transform as trans

    transform = trans.SimilarityTransform()
    if not transform.estimate(
            np.asarray(landmarks, dtype=np.float32), _ALIGNMENT_TEMPLATE):
        raise ValueError("Could not estimate IJB alignment transform")
    aligned = cv2.warpAffine(
        image, transform.params[:2, :], (112, 112), borderValue=0.0)
    aligned = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
    pair = np.stack((aligned, np.fliplr(aligned)))
    pair = pair.transpose(0, 3, 1, 2).astype(np.float32)
    return np.ascontiguousarray(pair / 127.5 - 1.0)


class IJBCSourceDataset(Dataset):
    """Return both deterministic orientations for each IJB source image."""

    def __init__(self, root, target="IJBC"):
        self.root = os.fspath(root)
        self.crop_root = os.path.join(self.root, "loose_crop")
        self.lines = _read_metadata(self.root, target)

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, source_index):
        fields = self.lines[int(source_index)].split()
        if len(fields) < 11:
            raise ValueError(f"Invalid IJB metadata row {source_index}")
        image = cv2.imread(os.path.join(self.crop_root, fields[0]))
        if image is None:
            raise FileNotFoundError(os.path.join(self.crop_root, fields[0]))
        landmarks = np.asarray(fields[1:11], dtype=np.float32).reshape(5, 2)
        pair = align_ijbc_pair(image, landmarks)
        return torch.from_numpy(pair), int(source_index)


class IJBCOrientationDataset(Dataset):
    """Select exact IJB source/orientation rows without stochastic transforms."""

    def __init__(self, root, orientations, target="IJBC"):
        self.sources = IJBCSourceDataset(root, target)
        self.orientations = tuple(
            (int(index), normalize_orientation(orientation))
            for index, orientation in orientations)
        if not self.orientations:
            raise ValueError("At least one IJB orientation is required")
        if any(index >= len(self.sources) for index, _ in self.orientations):
            raise IndexError("IJB replay source index is outside metadata")

    def __len__(self):
        return len(self.orientations)

    def __getitem__(self, item):
        source_index, orientation = self.orientations[int(item)]
        pair, _ = self.sources[source_index]
        return pair[orientation], source_index, orientation

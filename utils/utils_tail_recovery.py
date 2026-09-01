"""Helpers for replaying exact-dataset polynomial activation tails."""

import json
import os


def load_fixed_tail_replay_indices(path):
    """Load ordered MS1M source indices from a hard-tail mining manifest."""
    path = os.fspath(path)
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Tail replay manifest must be an object: {path}")
    indices = payload.get(
        "combined_source_indices", payload.get("dataset_indices"))
    if (not isinstance(indices, list) or not indices
            or any(type(index) is not int or index < 0 for index in indices)):
        raise ValueError(
            f"Tail replay manifest has invalid source indices: {path}")
    unique = tuple(dict.fromkeys(indices))
    if not unique:
        raise ValueError(f"Tail replay manifest is empty: {path}")
    return unique


def load_fixed_tail_replay_orientations(path, key="output_nonfinite"):
    """Load exact ``(source_index, orientation)`` rows from a manifest."""
    path = os.fspath(path)
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    rows = payload.get(key) if isinstance(payload, dict) else None
    if not isinstance(rows, list) or not rows:
        raise ValueError(
            f"Tail replay manifest has no non-empty {key!r} rows: {path}")
    orientations = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(
                f"Tail replay manifest has invalid {key!r} row: {path}")
        source_index = row.get("source_index")
        orientation = row.get("orientation")
        if (type(source_index) is not int or source_index < 0
                or type(orientation) is not int
                or orientation not in (0, 1)):
            raise ValueError(
                f"Tail replay manifest has invalid {key!r} row: {path}")
        orientations.append((source_index, orientation))
    return tuple(dict.fromkeys(orientations))

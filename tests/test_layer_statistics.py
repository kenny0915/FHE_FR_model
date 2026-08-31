import csv
import json

import numpy as np
import pytest
import torch
from torch import nn

from eval.layer_statistics import (
    LayerStatisticsRecorder,
    RunningTensorStats,
    evenly_spaced_indices,
    estimate_similarity_transform,
    parameter_rows,
    write_results,
)
from eval.compare_activation_distributions import compare_rows


def test_running_tensor_stats_aggregates_calls_and_nonfinite_values():
    stats = RunningTensorStats(
        "feature_map", "layer", "output", "Toy", sample_limit=3)
    stats.update(torch.tensor([[0.0, 1.0, float("nan")]]), dynamic_batch=True)
    stats.update(torch.tensor([[2.0, float("inf"), -1.0]]), dynamic_batch=True)

    row = stats.as_row()
    assert row["shape"] == "Nx3"
    assert row["calls"] == 2
    assert row["numel"] == 6
    assert row["finite"] == 4
    assert row["nan"] == 1
    assert row["posinf"] == 1
    assert row["mean"] == 0.5
    assert row["min"] == -1.0
    assert row["max"] == 2.0
    assert len(row["sample_values"]) == 3


def test_evenly_spaced_indices_stay_bounded_above_fp32_exact_range():
    indices = evenly_spaced_indices(80_281_600, 2048)

    assert indices.dtype == torch.long
    assert int(indices[0]) == 0
    assert int(indices[-1]) == 80_281_599
    assert bool((indices[1:] >= indices[:-1]).all())


def test_recorder_collects_leaf_outputs_and_parameters():
    model = nn.Sequential(nn.Linear(2, 2, bias=True), nn.ReLU())
    recorder = LayerStatisticsRecorder(model, sample_limit=8)
    model(torch.tensor([[1.0, -1.0], [2.0, 3.0]]))
    recorder.close()

    feature_rows = recorder.rows()
    assert {(row["layer"], row["tensor"]) for row in feature_rows} == {
        ("0", "output"),
        ("1", "output"),
    }
    assert all(row["calls"] == 1 for row in feature_rows)
    assert all(row["shape"] == "Nx2" for row in feature_rows)

    weights = parameter_rows(model, sample_limit=8)
    assert {(row["layer"], row["tensor"]) for row in weights} == {
        ("0", "weight"),
        ("0", "bias"),
    }


def test_write_results_creates_csv_and_json(tmp_path):
    stats = RunningTensorStats(
        "parameter", "layer", "weight", "Linear", sample_limit=0)
    stats.update(torch.tensor([1.0, 2.0, 3.0]))
    rows = [stats.as_row()]

    csv_path, json_path = write_results(
        rows, tmp_path / "report", {"source_images": 3})

    with csv_path.open() as handle:
        csv_rows = list(csv.DictReader(handle))
    with json_path.open() as handle:
        result = json.load(handle)
    assert len(csv_rows) == 1
    assert json.loads(csv_rows[0]["sample_values"]) == rows[0]["sample_values"]
    assert result["metadata"]["source_images"] == 3
    assert result["statistics"][0]["layer"] == "layer"
    assert result["statistics"][0]["abs_p99"] is None


def test_estimate_similarity_transform_recovers_known_mapping():
    source = np.array([
        [1.0, 2.0],
        [4.0, 2.0],
        [1.0, 5.0],
        [3.0, 4.0],
    ], dtype=np.float32)
    angle = np.deg2rad(25.0)
    rotation = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)],
    ])
    destination = (1.7 * (rotation @ source.T)).T + [8.0, -3.0]

    matrix = estimate_similarity_transform(source, destination)
    homogeneous_source = np.column_stack((source, np.ones(len(source))))

    np.testing.assert_allclose(
        (matrix @ homogeneous_source.T).T, destination, atol=1e-5)


def test_estimate_similarity_transform_rejects_degenerate_points():
    points = np.ones((5, 2), dtype=np.float32)

    with pytest.raises(ValueError, match="degenerate"):
        estimate_similarity_transform(points, points)


def test_compare_activation_rows_reports_tail_ratios_and_nonfinite():
    baseline = {("prelu", "input"): {
        "std": 2.0, "abs_p99": 4.0, "abs_p999": 5.0,
        "absmax": 8.0, "nonfinite": 0,
    }}
    candidate = {("prelu", "input"): {
        "std": 1.0, "abs_p99": 2.0, "abs_p999": 10.0,
        "absmax": 32.0, "nonfinite": 3,
    }}

    row = compare_rows(baseline, candidate)[0]

    assert row["std_ratio"] == pytest.approx(0.5)
    assert row["abs_p99_ratio"] == pytest.approx(0.5)
    assert row["abs_p999_ratio"] == pytest.approx(2.0)
    assert row["absmax_ratio"] == pytest.approx(4.0)
    assert row["candidate_nonfinite"] == 3

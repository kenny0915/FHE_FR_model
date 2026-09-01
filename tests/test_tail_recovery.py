import json

import pytest
import torch

from mine_herpn_tails import merge_rank_payloads, update_tail_heap
from utils.utils_tail_recovery import load_fixed_tail_replay_indices


def test_tail_heap_keeps_largest_source_orientation_rows():
    heap = []
    update_tail_heap(
        heap,
        torch.tensor([1.0, 5.0, 3.0, 9.0]),
        [10, 11, 12, 13],
        [0, 1, 0, 1],
        2,
    )
    assert sorted(heap, reverse=True) == [(9.0, 13, 1), (5.0, 11, 1)]


def test_rank_payload_merge_round_robins_activations(tmp_path):
    payload = {
        "output_nonfinite": [{"source_index": 9, "orientation": 1}],
        "activations": {
            "a": {
                "nonfinite_input_count": 0,
                "tail": [
                    {"source_index": 1, "orientation": 0, "absmax": 10.0},
                    {"source_index": 2, "orientation": 0, "absmax": 9.0},
                ],
            },
            "b": {
                "nonfinite_input_count": 1,
                "tail": [
                    {"source_index": 3, "orientation": 1, "absmax": 100.0},
                    {"source_index": 1, "orientation": 1, "absmax": 90.0},
                ],
            },
        },
    }
    merged = merge_rank_payloads([payload], ("a", "b"), 2)
    assert merged["combined_source_indices"] == [1, 3, 2]
    assert merged["activations"]["b"]["nonfinite_input_count"] == 1

    path = tmp_path / "tails.json"
    path.write_text(json.dumps(merged))
    assert load_fixed_tail_replay_indices(path) == (1, 3, 2)


def test_fixed_tail_manifest_rejects_invalid_indices(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"combined_source_indices": [1, -2]}))
    with pytest.raises(ValueError, match="invalid source indices"):
        load_fixed_tail_replay_indices(path)

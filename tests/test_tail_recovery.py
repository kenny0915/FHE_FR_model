import json

import pytest
import torch

from mine_herpn_tails import merge_rank_payloads, update_tail_heap
from configs.ms1mv3_r50_no_relu_phase2_hard_tail_recovery import (
    config as hard_tail_config,
)
from configs.ms1mv3_r50_no_relu_phase2_joint_recovery import (
    config as joint_recovery_config,
)
from configs.ms1mv3_r50_no_relu_phase2_joint_grouped_recovery import (
    config as grouped_recovery_config,
)
from utils.utils_tail_recovery import (
    load_fixed_tail_replay_indices,
    load_fixed_tail_replay_orientations,
)


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
    assert merged["combined_source_indices"] == [9, 1, 3, 2]
    assert merged["exact_nonfinite_source_count"] == 1
    assert merged["activations"]["b"]["nonfinite_input_count"] == 1

    path = tmp_path / "tails.json"
    path.write_text(json.dumps(merged))
    assert load_fixed_tail_replay_indices(path) == (9, 1, 3, 2)
    assert load_fixed_tail_replay_orientations(path) == ((9, 1),)


def test_fixed_tail_manifest_rejects_invalid_indices(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"combined_source_indices": [1, -2]}))
    with pytest.raises(ValueError, match="invalid source indices"):
        load_fixed_tail_replay_indices(path)


def test_fixed_tail_manifest_rejects_invalid_orientations(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({
        "output_nonfinite": [{"source_index": 1, "orientation": 2}],
    }))
    with pytest.raises(ValueError, match="invalid 'output_nonfinite' row"):
        load_fixed_tail_replay_orientations(path)


def test_hard_tail_recovery_covers_stem_through_layer3_only():
    names = hard_tail_config.herpn_range_loss_names
    assert len(names) == 22
    assert names[0] == "prelu"
    assert "layer1.0.prelu" in names
    assert "layer2.3.prelu" in names
    assert "layer3.13.prelu" in names
    assert all(not name.startswith("layer4") for name in names)
    assert hard_tail_config.freeze_batchnorm_running_stats
    assert hard_tail_config.freeze_batchnorm_affine


def test_joint_recovery_trains_all_convs_and_all_25_polynomials():
    names = joint_recovery_config.herpn_range_loss_names
    prefixes = joint_recovery_config.backbone_trainable_prefixes
    assert len(names) == 25
    assert names[0] == "prelu"
    assert names[-1] == "layer4.2.prelu"
    assert "conv1" in prefixes
    assert "layer4.2.conv2" in prefixes
    assert "layer4.2.prelu.herpn" in prefixes
    assert joint_recovery_config.herpn_independent_basis_scales
    assert joint_recovery_config.herpn_basis_anchor_loss_weight > 0.0
    assert joint_recovery_config.freeze_batchnorm_running_stats
    assert joint_recovery_config.freeze_batchnorm_affine
    assert joint_recovery_config.fixed_tail_replay_orientations_key == (
        "output_nonfinite")


def test_grouped_joint_recovery_separates_optimizer_scale_and_clipping():
    assert (grouped_recovery_config.backbone_trainable_prefixes
            == joint_recovery_config.backbone_trainable_prefixes)
    assert (grouped_recovery_config.herpn_range_loss_names
            == joint_recovery_config.herpn_range_loss_names)
    assert grouped_recovery_config.split_conv_herpn_optimizer
    assert grouped_recovery_config.separate_conv_herpn_gradient_clip
    assert grouped_recovery_config.lr == pytest.approx(1e-6)
    assert grouped_recovery_config.herpn_lr_multiplier == pytest.approx(10.0)
    assert grouped_recovery_config.conv_gradient_clip == pytest.approx(1.0)
    assert grouped_recovery_config.herpn_gradient_clip == pytest.approx(0.1)
    assert grouped_recovery_config.freeze_batchnorm_running_stats
    assert grouped_recovery_config.freeze_batchnorm_affine

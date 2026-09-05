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
from configs.ms1mv3_r50_no_relu_phase2_joint_tensor_recovery import (
    config as tensor_recovery_config,
)
from configs.ms1mv3_r50_no_relu_phase1_epoch24_plus3 import (
    config as phase1_plus3_config,
)
from configs.ms1mv3_r50_no_relu_phase2_conflict_aware_recovery import (
    config as conflict_recovery_config,
)
from configs.ms1mv3_r50_no_relu_phase2_ijbc_numerical_calibration import (
    config as ijbc_calibration_config,
)
from configs.ms1mv3_r50_no_relu_phase2_ijbc_numerical_focus1 import (
    config as ijbc_focus1_config,
)
from configs.ms1mv3_r50_no_relu_phase2_ms1mv3_numerical_calibration import (
    config as ms1mv3_calibration_config,
)
from configs.ms1mv3_r50_no_relu_phase2_ms1mv3_numerical_focus1 import (
    config as ms1mv3_focus1_config,
)
from configs.ms1mv3_r50_no_relu_phase2_ms1mv3_robust_margin import (
    config as ms1mv3_robust_margin_config,
)
from configs.ms1mv3_r50_no_relu_phase2_ms1mv3_robust_margin_focus1 import (
    config as ms1mv3_robust_margin_focus1_config,
)
from backbones.iresnet_no_relu import iresnet18
from train_ijbc_numerical_calibration import make_adversarial_tail_images
from utils.utils_multi_objective import (
    combine_conflict_aware_gradients,
    project_to_relative_trust_region,
)
from utils.utils_tail_recovery import (
    load_fixed_tail_replay_indices,
    load_fixed_tail_replay_orientations,
)
from utils.utils_ijbc_replay import load_ijbc_replay_orientations
from mine_ijbc_herpn_tails import merge_payloads as merge_ijbc_payloads


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


def test_ijbc_replay_merges_exact_csv_and_per_activation_json(tmp_path):
    csv_path = tmp_path / "nonfinite.csv"
    csv_path.write_text(
        "source_index,orientation\n9,flip\n10,original\n",
        encoding="utf-8")
    json_path = tmp_path / "tails.json"
    json_path.write_text(json.dumps({
        "output_nonfinite": [
            {"source_index": 9, "orientation": 1},
        ],
        "range_violations": [
            {"source_index": 13, "orientation": 1,
             "first_activation": "prelu", "input_absmax": 4.2},
        ],
        "activations": {
            "prelu": {"tail": [
                {"source_index": 11, "orientation": 0, "absmax": 7.0},
                {"source_index": 12, "orientation": 1, "absmax": 6.5},
            ]},
        },
    }), encoding="utf-8")
    assert load_ijbc_replay_orientations(
        (csv_path, json_path), activation_topk=1) == (
            (9, 1), (10, 0), (13, 1), (11, 0))


def test_ijbc_tail_merge_keeps_global_topk_and_failures():
    def payload(failure, value, index):
        return {
            "output_nonfinite": [
                {"source_index": failure, "orientation": 1}],
            "activations": {"prelu": {
                "nonfinite_input_count": failure,
                "tail": [{"source_index": index, "orientation": 0,
                          "absmax": value}],
            }},
        }
    merged = merge_ijbc_payloads(
        (payload(2, 8.0, 20), payload(3, 9.0, 30)), ("prelu",), 1)
    assert merged["output_nonfinite"] == [
        {"source_index": 2, "orientation": 1},
        {"source_index": 3, "orientation": 1},
    ]
    assert merged["activations"]["prelu"]["tail"][0]["source_index"] == 30
    assert merged["activations"]["prelu"]["nonfinite_input_count"] == 5


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


def test_tensor_joint_recovery_updates_every_tensor_with_sparse_replay():
    assert (tensor_recovery_config.backbone_trainable_prefixes
            == (*joint_recovery_config.backbone_trainable_prefixes, "fc"))
    assert tensor_recovery_config.optimizer == "adamw"
    assert tensor_recovery_config.lr == pytest.approx(1e-7)
    assert tensor_recovery_config.herpn_lr_multiplier == pytest.approx(10.0)
    assert (tensor_recovery_config.conv_herpn_gradient_clip_granularity
            == "tensor")
    assert tensor_recovery_config.other_backbone_gradient_clip == pytest.approx(
        1.0)
    assert tensor_recovery_config.fixed_tail_replay_batch_size == 4
    assert tensor_recovery_config.fixed_tail_replay_interval == 8
    average_tail_fraction = (
        tensor_recovery_config.fixed_tail_replay_batch_size
        / tensor_recovery_config.batch_size
        / tensor_recovery_config.fixed_tail_replay_interval)
    assert average_tail_fraction == pytest.approx(0.00390625)
    assert tensor_recovery_config.freeze_batchnorm_running_stats
    assert tensor_recovery_config.freeze_batchnorm_affine


def test_phase1_plus3_resumes_epoch24_optimizer_into_epoch27():
    assert phase1_plus3_config.resume
    assert phase1_plus3_config.resume_optimizer_state
    assert phase1_plus3_config.resume_rebase_lr_scheduler
    assert phase1_plus3_config.lr == pytest.approx(1e-4)
    assert phase1_plus3_config.num_epoch == 27
    assert phase1_plus3_config.resume_checkpoint_dir.endswith(
        "herpn_full_conversion_phase1")
    assert phase1_plus3_config.output != (
        phase1_plus3_config.resume_checkpoint_dir)


def test_conflict_recovery_keeps_exact_quadratic_and_separates_bn_policy():
    assert conflict_recovery_config.backbone_init.endswith(
        "model_epoch_23.pt")
    assert conflict_recovery_config.herpn_range_limit == pytest.approx(6.0)
    assert conflict_recovery_config.herpn_range_guard_ratio == pytest.approx(
        0.8)
    assert conflict_recovery_config.herpn_range_penalty_mode == (
        "sample_max_tail")
    assert conflict_recovery_config.herpn_training_stabilization_limit == (
        pytest.approx(6.0))
    assert conflict_recovery_config.optimizer == "sgd"
    assert conflict_recovery_config.momentum == pytest.approx(0.0)
    assert conflict_recovery_config.parameter_trust_region_ratio == (
        pytest.approx(0.005))
    assert conflict_recovery_config.parameter_trust_region_interval == 100
    assert conflict_recovery_config.num_epoch == 3


def test_ijbc_calibration_replays_latest_failures_with_fixed_bn_graph():
    assert ijbc_calibration_config.backbone_init.endswith("model_epoch_23.pt")
    assert ijbc_calibration_config.herpn_range_limit == pytest.approx(6.0)
    assert ijbc_calibration_config.herpn_range_guard_ratio == pytest.approx(0.75)
    assert ijbc_calibration_config.ijbc_gate_failure_repeats == 16
    assert ijbc_calibration_config.causal_range_reduction == "mean_max"
    assert ijbc_calibration_config.parameter_trust_region_ratio <= 0.01
    assert ijbc_focus1_config.backbone_init.endswith("model_epoch_05.pt")
    assert ijbc_focus1_config.ijbc_gate_failure_repeats == 64
    assert ijbc_focus1_config.ijbc_priority_manifests[0].endswith(
        "full_gate_epoch_05.json")
    assert ms1mv3_calibration_config.calibration_dataset == "ms1mv3"
    assert ms1mv3_calibration_config.backbone_init.endswith(
        "model_epoch_23.pt")
    assert (ms1mv3_calibration_config.calibration_priority_manifests
            == ms1mv3_calibration_config.calibration_replay_manifests)
    assert ms1mv3_focus1_config.backbone_init.endswith("model_epoch_04.pt")
    assert ms1mv3_focus1_config.ijbc_gate_failure_repeats == 64
    assert ms1mv3_focus1_config.calibration_priority_manifests[0].endswith(
        "full_gate_epoch_04.json")


def test_ms1mv3_robust_margin_uses_wide_replay_and_bounded_attack():
    config = ms1mv3_robust_margin_config
    assert config.backbone_init.endswith("model_epoch_23.pt")
    assert config.calibration_replay_activation_topk == 4096
    assert config.herpn_range_limit == pytest.approx(6.0)
    assert config.herpn_range_guard_ratio == pytest.approx(2.0 / 3.0)
    assert config.numerical_range_gate_limit == pytest.approx(4.0)
    assert config.adversarial_tail_enabled
    assert config.adversarial_tail_epsilon == pytest.approx(16.0 / 255.0)
    assert config.adversarial_tail_steps == 3
    focus = ms1mv3_robust_margin_focus1_config
    assert focus.backbone_init.endswith("model_epoch_03.pt")
    assert focus.calibration_priority_manifests[0].endswith(
        "full_gate_epoch_03.json")
    assert focus.calibration_replay_activation_topk == 64
    assert focus.ijbc_gate_failure_repeats == 256
    assert focus.numerical_range_gate_limit == pytest.approx(4.0)


def test_causal_range_penalty_uses_earliest_violation_per_sample():
    model = iresnet18(
        herpn_progress=5.0,
        herpn_range_penalty_mode="sample_max_tail",
    )
    activations = dict(model.named_modules())
    names = ("layer1.0.prelu", "layer1.1.prelu", "layer2.0.prelu")
    activations[names[0]]._last_sample_range_penalty = torch.tensor([0.2, 0.0])
    activations[names[1]]._last_sample_range_penalty = torch.tensor([9.0, 0.3])
    activations[names[2]]._last_sample_range_penalty = torch.tensor([99.0, 7.0])
    # Sample zero selects 0.2 from the first layer; sample one selects 0.3
    # from the second.  Later explosions cannot dominate either sample.
    assert model.herpn_causal_range_penalty(names).item() == pytest.approx(0.3)
    assert model.herpn_causal_range_penalty(
        names, reduction="mean").item() == pytest.approx(0.25)
    assert model.herpn_causal_range_penalty(
        names, reduction="mean_max").item() == pytest.approx(0.28)
    with pytest.raises(ValueError, match="causal range reduction"):
        model.herpn_causal_range_penalty(names, reduction="sum")


def test_adversarial_range_objective_has_gradient_below_guard():
    model = iresnet18(
        herpn_progress=5.0,
        herpn_range_penalty_mode="sample_max_tail",
    )
    activations = dict(model.named_modules())
    names = ("layer1.0.prelu", "layer1.1.prelu")
    first = torch.tensor([0.2, 0.4], requires_grad=True)
    second = torch.tensor([0.3, 0.1], requires_grad=True)
    activations[names[0]]._last_sample_input_ratio = first
    activations[names[1]]._last_sample_input_ratio = second
    objective = model.herpn_adversarial_range_objective(names, reduction="mean")
    expected = torch.log1p(torch.tensor([0.3, 0.4])).mean()
    assert objective.item() == pytest.approx(expected.item())
    objective.backward()
    assert first.grad is not None
    assert second.grad is not None
    assert first.grad[1] > 0
    assert second.grad[0] > 0


def test_adversarial_tail_images_respect_pixel_and_epsilon_bounds():
    class ToyRangeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.ones(()))
            self.ratio = None

        def forward(self, images):
            self.ratio = images.square().flatten(1).mean(dim=1) * self.scale
            return images.flatten(1)

        def herpn_adversarial_range_objective(self, names, reduction):
            assert names == ("toy",)
            assert reduction == "mean_max"
            values = torch.log1p(self.ratio)
            return values.mean() + 0.1 * values.amax()

    model = ToyRangeModel()
    images = torch.full((2, 3, 4, 4), 0.25)
    adversarial, objective = make_adversarial_tail_images(
        model, images, ("toy",), epsilon=0.1, step_size=0.05,
        steps=2, random_start=False)
    assert torch.all(adversarial <= images + 0.1 + 1e-7)
    assert torch.all(adversarial >= images - 0.1 - 1e-7)
    assert torch.all(adversarial <= 1.0)
    assert torch.all(adversarial >= -1.0)
    assert objective.item() > 0.0
    assert not torch.equal(adversarial, images)


def test_conflict_gradient_projection_and_trust_region_are_bounded():
    parameter = torch.nn.Parameter(torch.tensor([1.0, 0.0]))
    clean = (torch.tensor([1.0, 0.0]),)
    tail = (torch.tensor([-2.0, 2.0]),)
    combined, stats = combine_conflict_aware_gradients(
        (parameter,), clean, tail,
        learning_rate=0.1,
        tail_to_clean_ratio=1.0,
        max_step_update_ratio=1.0,
        scale_floor=1.0,
    )
    assert stats["conflicts"] == 1
    assert torch.dot(combined[0] - clean[0], clean[0]) >= 0

    anchor = parameter.detach().clone()
    with torch.no_grad():
        parameter.add_(torch.tensor([2.0, 0.0]))
    trust = project_to_relative_trust_region(
        (parameter,), (anchor,), ratio=0.01, scale_floor=1.0)
    assert trust["projected"] == 1
    assert torch.linalg.vector_norm(parameter - anchor).item() == pytest.approx(
        0.01)

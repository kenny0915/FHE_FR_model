import copy
from collections import OrderedDict

import pytest
import torch
from torch.nn import functional as F

from controlled_degree2.calibrate import reference_ranges, weighted_quadratic_abs_fit
from controlled_degree2.augment import prepare_range_batch
from controlled_degree2.mine_deployment_tails import merge_rank_payloads
from controlled_degree2.model import (
    DirectQuadratic,
    collect_causal_tail_penalty,
    damp_quadratic_terms,
    prelu_to_quadratic_coefficients,
    scale_intervals,
    set_quadratic_schedule,
)
from controlled_degree2.train import (
    belongs_to_frozen_module,
    causal_tail_names,
    deployment_tail_penalty,
    freeze_through_layer3,
    freeze_through_layer4,
    keep_batchnorm_eval,
    keep_frozen_modules_eval,
    make_adversarial_tail_batch,
    prioritized_deployment_rows,
    restore_batchnorm_state,
    snapshot_batchnorm_state,
)


def test_uniform_degree2_fit_has_expected_closed_form():
    slope = torch.tensor([0.0, 0.25])
    coefficients = prelu_to_quadratic_coefficients(slope, [2.0, 4.0])

    expected_even = torch.tensor([[3.0 / 8.0, 15.0 / 32.0],
                                  [3.0 / 4.0, 15.0 / 64.0]])
    expected = torch.stack(
        ((1 - slope) / 2 * expected_even[:, 0],
         (1 + slope) / 2,
         (1 - slope) / 2 * expected_even[:, 1]),
        dim=1,
    )
    assert torch.allclose(coefficients, expected)


def test_training_alpha_zero_is_exact_prelu_and_collects_hinge():
    slopes = torch.tensor([0.1, 0.4])
    activation = DirectQuadratic(2, lam_fit=[2.0, 3.0], lam_reg=[1.0, 1.5], slope=slopes)
    activation.alpha = 0.0
    activation.train()
    inputs = torch.tensor([[[[-2.0]], [[2.0]]]], requires_grad=True)

    actual = activation(inputs)
    expected = F.prelu(inputs, slopes)

    assert torch.equal(actual, expected)
    assert activation.last_oor == pytest.approx(1.0)
    assert activation.last_penalty.item() > 0


def test_causal_tail_penalty_is_bounded_and_differentiable_past_cap():
    activation = DirectQuadratic(1, lam_fit=2.0, lam_reg=1.0, slope=0.25, name="act")
    set_quadratic_schedule(activation, causal_guard_ratio=0.5)
    activation.train()
    inputs = torch.tensor([[[[100.0]]]], requires_grad=True)

    activation(inputs)
    penalty = collect_causal_tail_penalty(activation, ["act"], guard_ratio=0.5)
    penalty.backward()

    assert penalty.item() == pytest.approx(32.0 ** 2)
    assert torch.isfinite(inputs.grad).all()
    assert inputs.grad.abs().item() > 0


def test_causal_tail_penalty_remains_finite_after_downstream_overflow():
    activation = DirectQuadratic(1, lam_fit=2.0, lam_reg=1.0, name="act")
    activation.train()
    activation(torch.tensor([[[[float("inf")]]]]))

    penalty = collect_causal_tail_penalty(activation, ["act"])

    assert penalty.item() == pytest.approx(32.0 ** 2)


def test_causal_guard_threshold_matches_escape_assignment_threshold():
    activation = DirectQuadratic(1, lam_fit=2.0, lam_reg=1.0, slope=0.25, name="act")
    set_quadratic_schedule(activation, causal_guard_ratio=0.5)
    activation.train()
    inputs = torch.tensor([[[[0.75]]]], requires_grad=True)

    activation(inputs)
    penalty = collect_causal_tail_penalty(activation, ["act"], guard_ratio=0.5)
    penalty.backward()

    assert penalty.item() == pytest.approx((0.75 - 0.5) ** 2)
    assert inputs.grad.item() > 0


def test_causal_tail_uses_per_sample_max_without_spatial_dilution():
    activation = DirectQuadratic(1, lam_fit=2.0, lam_reg=1.0, slope=0.25, name="act")
    set_quadratic_schedule(activation, causal_guard_ratio=1.0)
    activation.train()
    inputs = torch.zeros(1, 1, 16, 16, requires_grad=True)
    with torch.no_grad():
        inputs[0, 0, 3, 7] = 2.0

    activation(inputs)
    penalty = collect_causal_tail_penalty(activation, ["act"], guard_ratio=1.0)

    assert penalty.item() == pytest.approx(1.0)


def test_adversarial_tail_search_increases_target_peak_without_parameter_grads():
    activation = DirectQuadratic(1, lam_fit=2.0, lam_reg=1.0, slope=0.25, name="act")
    model = torch.nn.Sequential(OrderedDict((("act", activation),))).train()
    set_quadratic_schedule(model, causal_guard_ratio=1.0)
    images = torch.full((4, 1, 4, 4), 0.1)
    eligible = torch.tensor([True, True, True, False])
    model(images[:1])
    before = float(activation.last_sample_peak.max())

    output, mask = make_adversarial_tail_batch(
        model,
        images,
        eligible,
        "act",
        fraction=0.5,
        steps=2,
        epsilon=0.4,
        step_size=0.2,
    )
    model(output[mask])
    after = float(activation.last_sample_peak.max())

    assert mask.sum().item() == 2
    assert not mask[-1]
    assert float(output.min()) >= -1.0
    assert float(output.max()) <= 1.0
    assert float((output - images).abs().max()) <= 0.4 + 1e-6
    assert after > before
    assert all(parameter.grad is None for parameter in model.parameters())


def test_deployment_shadow_is_unclipped_uses_eval_bn_and_restores_flags():
    activation = DirectQuadratic(
        1, lam_fit=1.0, lam_reg=1.0, slope=0.25, name="act"
    )
    model = torch.nn.Sequential(OrderedDict((
        ("bn", torch.nn.BatchNorm2d(1)),
        ("act", activation),
        ("pool", torch.nn.AdaptiveAvgPool2d(1)),
        ("flatten", torch.nn.Flatten()),
    ))).train()
    model.bn.running_mean.fill_(0.0)
    model.bn.running_var.fill_(1.0)
    before_batches = model.bn.num_batches_tracked.clone()
    inputs = torch.full((2, 1, 4, 4), 3.0)

    penalty, nonfinite, peak = deployment_tail_penalty(
        model, inputs, ["act"], guard_ratio=0.8
    )
    penalty.backward()

    assert penalty.item() > 0
    assert nonfinite == 0
    assert peak > 1.0
    assert model.bn.training
    assert activation.clip
    assert activation.causal_guard_ratio == pytest.approx(1.0)
    assert activation.tail_gradient_clip == pytest.approx(0.0)
    assert torch.equal(model.bn.num_batches_tracked, before_batches)
    assert model.bn.weight.grad is not None


def test_deployment_shadow_clips_backward_signal_at_polynomial_inputs():
    activation = DirectQuadratic(
        1, lam_fit=1.0, lam_reg=1.0, slope=0.25, name="act"
    )
    model = torch.nn.Sequential(OrderedDict((
        ("amplify", torch.nn.Conv2d(1, 1, 1, bias=False)),
        ("act", activation),
        ("pool", torch.nn.AdaptiveAvgPool2d(1)),
        ("flatten", torch.nn.Flatten()),
    ))).train()
    model.amplify.weight.data.fill_(2.0)
    inputs = torch.ones(2, 1, 2, 2, requires_grad=True)

    penalty, nonfinite, peak = deployment_tail_penalty(
        model,
        inputs,
        ["act"],
        guard_ratio=0.8,
        assignment_ratio=1.0,
        gradient_clip=0.01,
    )
    penalty.backward()

    assert penalty.item() > 0
    assert nonfinite == 0
    assert peak > 1.0
    assert torch.isfinite(inputs.grad).all()
    assert inputs.grad.abs().max().item() <= 0.0200001
    assert torch.isfinite(model.amplify.weight.grad).all()
    assert activation.tail_gradient_clip == pytest.approx(0.0)


def test_deployment_shadow_assigns_true_escape_before_applying_margin():
    first = DirectQuadratic(1, lam_fit=1.0, lam_reg=1.0, name="first")
    second = DirectQuadratic(1, lam_fit=1.0, lam_reg=1.0, name="second")

    class Double(torch.nn.Module):
        def forward(self, values):
            return values * 2.0

    model = torch.nn.Sequential(OrderedDict((
        ("first", first),
        ("amplify", Double()),
        ("second", second),
        ("pool", torch.nn.AdaptiveAvgPool2d(1)),
        ("flatten", torch.nn.Flatten()),
    ))).train()
    inputs = torch.full((2, 1, 2, 2), 0.9, requires_grad=True)

    penalty, nonfinite, peak = deployment_tail_penalty(
        model,
        inputs,
        ["first", "second"],
        guard_ratio=0.8,
        assignment_ratio=1.0,
    )
    penalty.backward()

    # The first row is above the 0.8 margin but not outside the true interval;
    # assignment must therefore reach the second activation's >1.0x input.
    assert penalty.item() > 0.5
    assert nonfinite == 0
    assert peak > 1.0
    assert inputs.grad.abs().sum().item() > 0


def test_deployment_tail_manifest_merge_is_layer_balanced_and_deduplicated():
    def payload(rank, rows_a, rows_b, nonfinite=()):
        return {
            "rank": rank,
            "output_nonfinite": [
                {"source_index": index, "orientation": orientation}
                for index, orientation in nonfinite
            ],
            "activations": {
                "a": {"nonfinite_input_count": 0, "tail": rows_a},
                "b": {"nonfinite_input_count": rank, "tail": rows_b},
            },
        }

    row = lambda index, orientation, ratio: {
        "source_index": index,
        "orientation": orientation,
        "ratio": ratio,
        "absmax": ratio * 2,
    }
    merged = merge_rank_payloads([
        payload(0, [row(1, 0, 4.0), row(2, 0, 3.0)],
                [row(3, 1, 8.0), row(1, 0, 2.0)], nonfinite=((9, 1),)),
        payload(1, [row(1, 0, 5.0), row(4, 1, 1.0)],
                [row(5, 0, 7.0)], nonfinite=((9, 1),)),
    ], ("a", "b"), topk=2)

    combined = [
        (row["source_index"], row["orientation"])
        for row in merged["combined_orientations"]
    ]
    assert combined == [(9, 1), (1, 0), (3, 1), (2, 0), (5, 0)]
    assert merged["activations"]["a"]["tail"][0]["ratio"] == pytest.approx(5.0)
    assert merged["activations"]["b"]["nonfinite_input_count"] == 1


def test_deployment_tail_priority_repeats_manifest_prefix_only():
    rows = ((9, 1), (7, 0), (5, 1), (3, 0))

    weighted = prioritized_deployment_rows(rows, priority_count=2, priority_repeats=3)

    assert weighted == (
        (9, 1), (7, 0),
        (9, 1), (7, 0),
        (9, 1), (7, 0),
        (5, 1), (3, 0),
    )


def test_eval_is_unclipped_but_optional_diagnostic_clip_is_bounded():
    activation = DirectQuadratic(1, lam_fit=2.0, lam_reg=1.2, slope=0.25).eval()
    inputs = torch.tensor([[[[20.0]]]])
    deployable = activation(inputs)
    activation.clip_eval = True
    diagnostic = activation(inputs)

    assert deployable.item() > diagnostic.item() * 5


def test_interval_scaling_preserves_rescaled_polynomial_shape():
    activation = DirectQuadratic(2, lam_fit=[2.0, 4.0], lam_reg=[1.2, 2.4], slope=[0.1, 0.3])
    original = copy.deepcopy(activation).eval()
    calibration = {
        "act": {
            "lam_fit": [2.0, 4.0],
            "lam_reg": [1.2, 2.4],
            "even_coeffs": [[0.3, 0.4], [0.5, 0.2]],
        }
    }
    activation.name = "act"
    touched = scale_intervals(activation, calibration, {"act": 1.5})
    inputs = torch.randn(3, 2, 4, 4)

    assert touched == {"act": 1.5}
    assert torch.allclose(
        activation.eval()(inputs), 1.5 * original(inputs / 1.5), rtol=1e-6, atol=1e-6
    )
    assert calibration["act"]["lam_fit"] == pytest.approx([3.0, 6.0])


def test_quadratic_tail_damping_changes_only_c2_and_records_fit_update():
    activation = DirectQuadratic(
        2, lam_fit=[2.0, 4.0], lam_reg=[1.2, 2.4], slope=[0.1, 0.3]
    )
    activation.name = "layer1.0.prelu"
    before_coeffs = activation.coeffs.detach().clone()
    before_fit = activation.lam_fit.detach().clone()
    before_reg = activation.lam_reg.detach().clone()
    calibration = {
        "layer1.0.prelu": {
            "lam_fit": [2.0, 4.0],
            "lam_reg": [1.2, 2.4],
            "even_coeffs": [[0.3, 0.4], [0.5, 0.2]],
        }
    }

    touched = damp_quadratic_terms(
        activation, calibration, {"layer1": 0.9}
    )

    assert touched == {"layer1.0.prelu": 0.9}
    assert torch.equal(activation.coeffs[:, :2], before_coeffs[:, :2])
    assert torch.allclose(activation.coeffs[:, 2], before_coeffs[:, 2] * 0.9)
    assert torch.equal(activation.lam_fit, before_fit)
    assert torch.equal(activation.lam_reg, before_reg)
    assert torch.allclose(
        torch.tensor(calibration["layer1.0.prelu"]["even_coeffs"]),
        torch.tensor([[0.3, 0.36], [0.5, 0.18]]),
    )
    assert calibration["layer1.0.prelu"]["quadratic_tail_scale"] == pytest.approx(0.9)


def test_histogram_weighted_fit_returns_a_finite_direct_quadratic():
    centers = torch.logspace(-3, 1, 512, dtype=torch.float64)
    widths = torch.diff(torch.cat((centers[:1] / 1.01, centers)))
    probabilities = torch.exp(-0.5 * (centers[None, :] / torch.tensor([[0.8], [1.4]])) ** 2)
    probabilities /= probabilities.sum(dim=1, keepdim=True)
    coefficients, errors = weighted_quadratic_abs_fit(
        probabilities, centers, widths, torch.tensor([3.0, 5.0]), fit_eps=0.05
    )

    assert coefficients.shape == (2, 2)
    assert errors.shape == (2,)
    assert torch.isfinite(coefficients).all()
    assert torch.isfinite(errors).all()
    assert bool((errors < 0.6).all())


def test_reference_uses_deployed_range_buffers_and_widening_metadata(tmp_path):
    path = tmp_path / "run10.pt"
    torch.save(
        {
            "format": "reference",
            "state_dict": {
                "prelu.lam_fit": torch.tensor([2.0, 3.0]),
                "prelu.lam_reg": torch.tensor([1.2, 1.8]),
            },
            "poly_calib": {
                "prelu": {
                    "lam_fit": [99.0, 99.0],
                    "lam_reg": [98.0, 98.0],
                    "lam_scale": 1.5,
                }
            },
        },
        path,
    )

    ranges, checkpoint_format = reference_ranges(path, ["prelu"], {"prelu": 2})
    lam_fit, lam_reg, scale = ranges["prelu"]

    assert checkpoint_format == "reference"
    assert lam_fit.tolist() == [2.0, 3.0]
    assert lam_reg.tolist() == pytest.approx([1.2, 1.8])
    assert scale == 1.5


def test_registered_controlled_backbone_has_25_direct_quadratics():
    from backbones import get_model

    model = get_model("r50_controlled_d2", dropout=0, fp16=False)
    activations = [module for module in model.modules() if isinstance(module, DirectQuadratic)]

    assert len(activations) == 25
    assert all(module.coeffs.shape[1] == 3 for module in activations)


def test_causal_scan_covers_all_trainable_polynomials_in_deployment_order():
    from backbones import get_model

    model = get_model("r50_controlled_d2", dropout=0, fp16=False)
    all_names = causal_tail_names(model)
    after_layer3_freeze = causal_tail_names(model, freeze_through_layer3(model))

    assert len(all_names) == 25
    assert all_names[0] == "prelu"
    assert all_names[-1] == "layer4.2.prelu"
    assert after_layer3_freeze == [
        "layer4.0.prelu",
        "layer4.1.prelu",
        "layer4.2.prelu",
    ]


def test_freeze_through_layer3_locks_parameters_and_batchnorm_state():
    from backbones import get_model

    model = get_model("r50_controlled_d2", dropout=0, fp16=False)
    model.train()
    frozen = freeze_through_layer3(model)

    assert belongs_to_frozen_module("layer3.13.prelu", frozen)
    assert not belongs_to_frozen_module("layer4.0.prelu", frozen)
    assert all(not parameter.requires_grad for parameter in model.layer3.parameters())
    assert any(parameter.requires_grad for parameter in model.layer4.parameters())
    assert not model.layer3.training

    model.train()
    keep_frozen_modules_eval(model, frozen)
    assert not model.layer3.training
    assert model.layer4.training


def test_freeze_through_layer4_leaves_only_embedding_head_trainable():
    from backbones import get_model

    model = get_model("r50_controlled_d2", dropout=0, fp16=False)
    frozen = freeze_through_layer4(model)

    assert belongs_to_frozen_module("layer4.2.prelu", frozen)
    assert all(not parameter.requires_grad for parameter in model.layer4.parameters())
    assert any(parameter.requires_grad for parameter in model.fc.parameters())
    assert not model.layer4.training


def test_keep_batchnorm_eval_preserves_trainable_affine_parameters():
    model = torch.nn.Sequential(
        torch.nn.Conv2d(2, 2, 1),
        torch.nn.BatchNorm2d(2),
        torch.nn.Sequential(torch.nn.BatchNorm1d(2)),
    ).train()

    keep_batchnorm_eval(model)

    batchnorms = [
        module
        for module in model.modules()
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
    ]
    assert batchnorms
    assert all(not module.training for module in batchnorms)
    assert all(module.weight.requires_grad for module in batchnorms)


def test_batchnorm_state_can_be_rolled_back_after_rejected_forward():
    model = torch.nn.Sequential(
        torch.nn.BatchNorm2d(2),
        torch.nn.Flatten(),
        torch.nn.Linear(8, 2),
        torch.nn.BatchNorm1d(2),
    ).train()
    before = snapshot_batchnorm_state(model)

    model(torch.full((4, 2, 2, 2), float("nan")))
    assert not torch.isfinite(model[0].running_mean).all()
    assert not torch.isfinite(model[3].running_var).all()

    restore_batchnorm_state(model, before)
    for name, expected in before.items():
        module = model.get_submodule(name)
        running_mean, running_var, num_batches_tracked = expected
        assert torch.equal(module.running_mean, running_mean)
        assert torch.equal(module.running_var, running_var)
        assert torch.equal(module.num_batches_tracked, num_batches_tracked)


def test_range_coverage_keeps_realistic_domain_and_masks_pathological_rows():
    torch.manual_seed(9)
    images = torch.rand(4, 3, 112, 112) * 2.0 - 1.0
    output, mask = prepare_range_batch(
        images,
        pathological_fraction=0.25,
        crop_probability=1.0,
        lowres_probability=1.0,
        photo_probability=1.0,
        stress_probability=1.0,
    )

    assert output.shape == images.shape
    assert torch.isfinite(output).all()
    assert float(output.min()) >= -1.0
    assert float(output.max()) <= 1.0
    assert mask.tolist() == [True, True, True, False]

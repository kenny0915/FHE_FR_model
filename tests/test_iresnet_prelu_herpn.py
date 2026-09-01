import importlib.util
import sys
import types

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from backbones import get_model
from backbones.iresnet_nl9_prelu_herpn import NL9_ACTIVATION_NAMES
from backbones.iresnet_prelu_herpn import (
    PReLUHerPNActivation,
    PReLULinearActivation,
)
from backbones.iresnet_no_relu import ProgressiveHerPNActivation
from utils.utils_optimizer import split_weight_decay_parameters


class _EasyDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


def _load_standalone_config(path):
    fake_easydict = types.ModuleType("easydict")
    fake_easydict.EasyDict = _EasyDict
    previous = sys.modules.get("easydict")
    sys.modules["easydict"] = fake_easydict
    try:
        spec = importlib.util.spec_from_file_location(
            "_test_prelu_herpn_config", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.config
    finally:
        if previous is None:
            del sys.modules["easydict"]
        else:
            sys.modules["easydict"] = previous


def test_baseline_prelu_state_loads_strictly_and_blend_zero_is_exact():
    activation = PReLUHerPNActivation(
        channels=3, blend=0.0).eval()
    slopes = torch.tensor([0.1, -0.2, 0.4])
    activation.load_state_dict({"weight": slopes}, strict=True)

    inputs = torch.randn(2, 3, 4, 4)
    expected = F.prelu(inputs, slopes)
    actual = activation(inputs)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(activation.prelu.weight, slopes)
    assert not activation.prelu.weight.requires_grad
    torch.testing.assert_close(
        activation.herpn.weight,
        torch.zeros_like(activation.herpn.weight))
    torch.testing.assert_close(
        activation.herpn.bias,
        torch.zeros_like(activation.herpn.bias))


def test_prelu_herpn_uses_exact_prelu_decomposition():
    torch.manual_seed(2)
    activation = PReLUHerPNActivation(
        channels=3, blend=1.0).eval()
    with torch.no_grad():
        activation.prelu.weight.copy_(
            torch.tensor([0.1, -0.2, 0.4]))
        activation.herpn.weight.uniform_(0.5, 1.3)
        activation.herpn.bias.uniform_(-0.2, 0.2)

    inputs = torch.randn(2, 3, 4, 4)
    slope = activation.prelu.weight.reshape(1, -1, 1, 1)
    expected = (
        slope * inputs
        + (1.0 - slope) * activation.herpn(inputs)
    )

    torch.testing.assert_close(
        activation(inputs), expected, rtol=1e-6, atol=1e-6)


def test_prelu_herpn_folds_to_the_same_degree_two_polynomial():
    torch.manual_seed(3)
    activation = PReLUHerPNActivation(
        channels=3, blend=1.0).eval()
    with torch.no_grad():
        activation.prelu.weight.copy_(
            torch.tensor([0.1, -0.2, 0.4]))
        activation.herpn.weight.uniform_(0.5, 1.3)
        activation.herpn.bias.uniform_(-0.2, 0.2)
        for batchnorm in (
                activation.herpn.bn0,
                activation.herpn.bn1,
                activation.herpn.bn2):
            batchnorm.running_mean.uniform_(-0.3, 0.3)
            batchnorm.running_var.uniform_(0.5, 1.5)

    folded = activation.folded().eval()
    inputs = torch.randn(2, 3, 4, 4)

    torch.testing.assert_close(
        folded(inputs), activation(inputs), rtol=1e-5, atol=1e-5)
    assert folded.coefficient2.shape == (3, 1, 1)


def test_layerwise_scaled_prelu_herpn_uses_normalized_hermite_input():
    torch.manual_seed(31)
    activation = PReLUHerPNActivation(
        channels=3, blend=0.0, layerwise_scale=True).eval()
    activation.set_input_scale(4.5)
    activation.set_blend(1.0)
    with torch.no_grad():
        activation.prelu.weight.copy_(torch.tensor([0.1, -0.2, 0.4]))
        activation.herpn.weight.uniform_(0.5, 1.3)
        activation.herpn.bias.uniform_(-0.2, 0.2)

    inputs = 3.0 * torch.randn(2, 3, 4, 4)
    scale = activation.input_scale
    slope = activation.prelu.weight.reshape(1, -1, 1, 1)
    expected = (
        slope * inputs
        + (1.0 - slope) * scale * activation.herpn(inputs / scale)
    )

    torch.testing.assert_close(
        activation(inputs), expected, rtol=1e-6, atol=1e-6)


def test_layerwise_scaled_prelu_herpn_folds_exactly():
    torch.manual_seed(32)
    activation = PReLUHerPNActivation(
        channels=3, blend=0.0, layerwise_scale=True).eval()
    activation.set_input_scale(7.25)
    activation.set_blend(1.0)
    with torch.no_grad():
        activation.prelu.weight.copy_(torch.tensor([0.1, -0.2, 0.4]))
        activation.herpn.weight.uniform_(0.5, 1.3)
        activation.herpn.bias.uniform_(-0.2, 0.2)
        for batchnorm in (
                activation.herpn.bn0,
                activation.herpn.bn1,
                activation.herpn.bn2):
            batchnorm.running_mean.uniform_(-0.3, 0.3)
            batchnorm.running_var.uniform_(0.5, 1.5)

    inputs = 5.0 * torch.randn(2, 3, 4, 4)
    folded = activation.folded().eval()
    torch.testing.assert_close(
        folded(inputs), activation(inputs), rtol=1e-5, atol=1e-5)


def test_prelu_herpn_polynomial_parameter_selection_is_group_local():
    model = get_model(
        "r18_prelu_herpn",
        dropout=0,
        fp16=False,
        herpn_progress=0.0,
        prelu_herpn_layerwise_scale=True,
    )
    activations = dict(model.named_progressive_activations())
    names = model.layerwise_poly_activation_names()

    selected = model.layerwise_poly_parameters(names[:2])
    expected = [
        parameter
        for name in names[:2]
        for parameter in (
            activations[name].herpn.weight,
            activations[name].herpn.bias,
        )
    ]

    assert {id(parameter) for parameter in selected} == {
        id(parameter) for parameter in expected}
    assert len(selected) == 4
    assert len(model.layerwise_poly_parameters()) == 2 * len(names)
    with pytest.raises(ValueError, match="Unknown PReLU-HerPN activations"):
        model.layerwise_poly_parameters(("missing.prelu",))


def test_prelu_linear_student_folds_exactly_without_a_square():
    activation = PReLULinearActivation(channels=3, blend=1.0).eval()
    with torch.no_grad():
        activation.weight.copy_(torch.tensor([[[0.8]], [[1.1]], [[0.3]]]))
        activation.bias.copy_(torch.tensor([[[0.2]], [[-0.1]], [[0.4]]]))
    inputs = torch.tensor([1.0e24, -1.0e24, 2.0]).reshape(1, 3, 1, 1)

    output = activation(inputs)
    folded = activation.folded().eval()

    assert torch.isfinite(output).all()
    torch.testing.assert_close(folded(inputs), output, rtol=0.0, atol=0.0)
    assert torch.count_nonzero(folded.coefficient2) == 0


def test_hybrid_linear_suffix_loads_legacy_checkpoint_at_exact_blend_zero():
    torch.manual_seed(33)
    source = get_model(
        "r18_no_relu", dropout=0, fp16=False, herpn_progress=0.0).eval()
    names = [
        name for name, module in source.named_modules()
        if isinstance(module, ProgressiveHerPNActivation)
    ]
    source.set_herpn_blends({
        name: float(index < 3) for index, name in enumerate(names)
    })
    hybrid = get_model(
        "r18_prelu_herpn",
        dropout=0,
        fp16=False,
        herpn_progress=0.0,
        prelu_herpn_layerwise_scale=True,
        prelu_herpn_legacy_prefix=3,
        prelu_herpn_linear_indices=(len(names) - 1,),
    ).eval()
    hybrid.load_backbone_init_state_dict(source.state_dict())

    activations = dict(hybrid.named_progressive_activations())
    assert isinstance(activations[names[-1]], PReLULinearActivation)
    selected = hybrid.layerwise_poly_parameters((names[-1],))
    assert {id(parameter) for parameter in selected} == {
        id(activations[names[-1]].weight),
        id(activations[names[-1]].bias),
    }
    inputs = torch.randn(1, 3, 112, 112)
    with torch.no_grad():
        expected = source(inputs)
        actual = hybrid(inputs)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_hybrid_linear_suffix_can_freeze_tail_safe_coefficients():
    model = get_model(
        "r18_prelu_herpn",
        dropout=0,
        fp16=False,
        herpn_progress=0.0,
        prelu_herpn_legacy_prefix=3,
        prelu_herpn_linear_indices=(8,),
        prelu_herpn_linear_trainable=False,
    )
    activation = model.progressive_activations()[8]

    assert isinstance(activation, PReLULinearActivation)
    assert not activation.prelu.weight.requires_grad
    assert not activation.weight.requires_grad
    assert not activation.bias.requires_grad


def test_scaled_checkpoint_restores_scaling_without_constructor_flag():
    source = PReLUHerPNActivation(
        channels=2, blend=0.0, layerwise_scale=True).eval()
    source.set_input_scale(5.0)
    source.set_blend(1.0)
    with torch.no_grad():
        source.herpn.weight.copy_(torch.tensor([[[0.7]], [[1.1]]]))
        source.herpn.bias.copy_(torch.tensor([[[0.2]], [[-0.1]]]))

    restored = PReLUHerPNActivation(channels=2, blend=0.0).eval()
    restored.load_state_dict(source.state_dict(), strict=True)
    inputs = torch.randn(2, 2, 4, 4)

    assert bool(restored.layerwise_scale_enabled)
    assert restored._scale_is_calibrated
    torch.testing.assert_close(
        restored(inputs), source(inputs), rtol=0.0, atol=0.0)


def test_layerwise_scaled_prelu_herpn_requires_calibration_before_blend():
    activation = PReLUHerPNActivation(
        channels=2, blend=0.0, layerwise_scale=True)

    with pytest.raises(RuntimeError, match="Calibrate"):
        activation.set_blend(0.1)

    activation.set_input_scale(3.0)
    activation.set_blend(0.1)
    assert activation._scale_is_calibrated
    assert float(activation.blend) == pytest.approx(0.1)


def test_layerwise_range_penalty_is_normalized_by_public_interval():
    activation = PReLUHerPNActivation(
        channels=1, blend=0.0, layerwise_scale=True).train()
    activation.set_input_scale(10.0)
    inputs = torch.tensor([[[[20.0, 5.0]]]])

    activation(inputs)

    # Relative excess is [1, 0]: mean square 0.5 plus 0.1 * max square.
    torch.testing.assert_close(
        activation.range_penalty(), torch.tensor(0.6))
    assert float(activation.range_stats()["outside_fraction"]) == pytest.approx(
        0.5)


def test_relative_distillation_remains_active_at_full_conversion():
    activation = PReLUHerPNActivation(
        channels=2, blend=1.0).train()
    inputs = torch.randn(4, 2, 3, 3)
    output = activation(inputs)
    loss = activation.distillation_loss()

    assert output.shape == inputs.shape
    assert loss is not None
    assert torch.isfinite(loss)
    assert float(loss) > 0.0
    loss.backward()
    assert activation.herpn.weight.grad is not None
    assert torch.isfinite(activation.herpn.weight.grad).all()
    assert activation.prelu.weight.grad is None


def test_distillation_gradient_is_local_but_task_gradient_reaches_input():
    activation = PReLUHerPNActivation(
        channels=2, blend=1.0).train()
    inputs = torch.randn(4, 2, 3, 3, requires_grad=True)
    output = activation(inputs)
    distillation_loss = activation.distillation_loss()
    distillation_loss.backward(retain_graph=True)

    assert activation.herpn.weight.grad is not None
    assert float(activation.herpn.weight.grad.abs().sum()) > 0.0
    assert inputs.grad is None or torch.count_nonzero(inputs.grad) == 0

    activation.zero_grad(set_to_none=True)
    inputs.grad = None
    output.square().mean().backward()
    assert inputs.grad is not None
    assert float(inputs.grad.abs().sum()) > 0.0


def test_hybrid_legacy_prefix_load_is_exact_and_new_students_reset():
    torch.manual_seed(41)
    source = get_model(
        "r18_no_relu", dropout=0, fp16=False, herpn_progress=0.0).eval()
    names = [
        name for name, module in source.named_modules()
        if isinstance(module, ProgressiveHerPNActivation)
    ]
    source.set_herpn_blends({
        name: float(index < 3) for index, name in enumerate(names)
    })
    source_state = source.state_dict()

    hybrid = get_model(
        "r18_prelu_herpn",
        dropout=0,
        fp16=False,
        herpn_progress=0.0,
        prelu_herpn_layerwise_scale=True,
        prelu_herpn_legacy_prefix=3,
    ).eval()
    hybrid.load_backbone_init_state_dict(source_state)
    hybrid.set_herpn_blends({
        name: float(index < 3) for index, name in enumerate(names)
    })

    activations = dict(hybrid.named_progressive_activations())
    assert all(isinstance(activations[name], ProgressiveHerPNActivation)
               for name in names[:3])
    assert all(isinstance(activations[name], PReLUHerPNActivation)
               for name in names[3:])
    assert hybrid.uncalibrated_layerwise_poly_names() == names[3:]
    for name in names[3:]:
        torch.testing.assert_close(
            activations[name].herpn.weight,
            torch.zeros_like(activations[name].herpn.weight))
        torch.testing.assert_close(
            activations[name].herpn.bias,
            torch.zeros_like(activations[name].herpn.bias))

    inputs = torch.randn(1, 3, 112, 112)
    with torch.no_grad():
        expected = source(inputs)
        actual = hybrid(inputs)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_hybrid_refuses_to_reinterpret_converted_legacy_suffix():
    source = get_model(
        "r18_no_relu", dropout=0, fp16=False, herpn_progress=0.0).eval()
    names = [
        name for name, module in source.named_modules()
        if isinstance(module, ProgressiveHerPNActivation)
    ]
    source.set_herpn_blends({
        name: float(index < 4) for index, name in enumerate(names)
    })
    hybrid = get_model(
        "r18_prelu_herpn",
        dropout=0,
        fp16=False,
        herpn_progress=0.0,
        prelu_herpn_layerwise_scale=True,
        prelu_herpn_legacy_prefix=3,
    )

    with pytest.raises(ValueError, match="already-converted legacy HerPN"):
        hybrid.load_backbone_init_state_dict(source.state_dict())


def test_epoch10_selective9_config_preserves_eight_and_targets_layer42():
    cfg = _load_standalone_config(
        "configs/ms1mv3_r50_herpn_epoch10_selective9_layer42.py")
    names = tuple(group[0] for group in cfg.herpn_conversion_groups)

    assert cfg.prelu_herpn_legacy_prefix == 8
    assert cfg.prelu_herpn_linear_indices == (24,)
    assert names[8] == "layer4.2.prelu"
    assert cfg.layerwise_poly_training_group_limit == 9
    assert cfg.layerwise_poly_preserve_batchnorm_during_local_fit
    assert cfg.layerwise_poly_preserve_batchnorm_during_blend
    assert cfg.layerwise_poly_preserve_batchnorm_during_final_finetune
    assert cfg.herpn_bn_recalibration_batches == 0
    assert not cfg.layerwise_poly_strict_recalibrate_before_blend
    assert not cfg.layerwise_poly_verify_singleton_boundary
    assert cfg.herpn_range_loss_weight == 0.0
    assert cfg.output.endswith("epoch10_linear9_layer42")
    assert not cfg.herpn_require_full_conversion

    resume_cfg = _load_standalone_config(
        "configs/ms1mv3_r50_herpn_epoch10_linear9_layer42_resume.py")
    assert resume_cfg.resume
    assert resume_cfg.output == cfg.output

    recovery_cfg = _load_standalone_config(
        "configs/"
        "ms1mv3_r50_herpn_epoch10_linear9_prelu_slope_recovery.py")
    assert not recovery_cfg.resume
    assert not recovery_cfg.prelu_herpn_linear_trainable
    assert recovery_cfg.herpn_group_epochs[8] == -2.0
    assert recovery_cfg.herpn_distill_loss_weight == 0.0
    assert recovery_cfg.task_loss_weight == 0.0
    assert recovery_cfg.backbone_trainable_prefixes[0] == "layer4.2.conv2"
    assert recovery_cfg.layerwise_poly_final_backbone_lr_scale == 1.0
    assert recovery_cfg.freeze_batchnorm_running_stats
    assert recovery_cfg.max_nonfinite_embedding_skips == 100
    assert recovery_cfg.output.endswith("linear9_prelu_slope_recovery")
    assert resume_cfg.herpn_conversion_groups == cfg.herpn_conversion_groups

    layer3_recovery = _load_standalone_config(
        "configs/"
        "ms1mv3_r50_herpn_epoch10_linear9_layer3_9_recovery.py")
    layer3_names = tuple(
        group[0] for group in layer3_recovery.herpn_conversion_groups)
    assert layer3_recovery.prelu_herpn_linear_indices == (17,)
    assert layer3_names[8] == "layer3.9.prelu"
    assert layer3_recovery.task_loss_weight == 0.0
    assert "layer3.10" in layer3_recovery.backbone_trainable_prefixes
    assert layer3_recovery.freeze_batchnorm_running_stats
    assert layer3_recovery.save_validation_snapshots

    low_lr = _load_standalone_config(
        "configs/"
        "ms1mv3_r50_herpn_epoch10_linear9_layer3_9_low_lr.py")
    assert low_lr.lr == 3.0e-5
    assert low_lr.verbose == 250
    assert low_lr.backbone_init.endswith("model_step_01000.pt")


def test_zero_initialized_student_has_bounded_relative_loss():
    activation = PReLUHerPNActivation(
        channels=3, blend=0.0).train()
    with torch.no_grad():
        activation.prelu.weight.copy_(
            torch.tensor([-0.1, -0.15, 0.05]))
    inputs = 0.1 * torch.randn(8, 3, 5, 5)
    activation(inputs)

    loss = activation.distillation_loss()
    assert torch.isfinite(loss)
    assert 0.0 < float(loss) < 2.0


def test_r50_config_schedules_every_prelu_herpn_activation():
    cfg = _load_standalone_config(
        "configs/ms1mv3_r50_prelu_herpn.py")
    model = get_model(
        cfg.network,
        dropout=0,
        fp16=False,
        herpn_range_limit=cfg.herpn_range_limit,
        herpn_bn_eps=cfg.herpn_bn_eps,
        herpn_progress=cfg.herpn_initial_progress,
        prelu_herpn_distill_eps=cfg.prelu_herpn_distill_eps,
    )
    expected = {
        name for name, module in model.named_modules()
        if isinstance(module, PReLUHerPNActivation)
    }
    scheduled = {
        name for group in cfg.herpn_conversion_groups for name in group
    }

    assert len(expected) == 25
    assert scheduled == expected
    assert sum(map(len, cfg.herpn_conversion_groups)) == 25
    assert len(cfg.herpn_group_epochs) == len(
        cfg.herpn_conversion_groups)
    assert all(
        right >= left + cfg.herpn_transition_epochs
        for left, right in zip(
            cfg.herpn_group_epochs, cfg.herpn_group_epochs[1:])
    )
    final_conversion_epoch = (
        cfg.herpn_group_epochs[-1] + cfg.herpn_transition_epochs)
    assert cfg.num_epoch - final_conversion_epoch == 4
    assert cfg.output == "work_dirs/ms1mv3_r50_prelu_herpn"


def test_legacy_r50_phase1_resume_finishes_all_conversions_safely():
    cfg = _load_standalone_config(
        "configs/ms1mv3_r50_no_relu_full_conversion_phase1.py")

    assert cfg.resume
    assert cfg.resume_checkpoint_dir == "work_dirs/ms1mv3_r50_herpn"
    assert cfg.output.endswith("herpn_full_conversion_phase1")
    assert cfg.output != cfg.resume_checkpoint_dir
    assert not cfg.resume_optimizer_state
    assert cfg.resume_rebase_lr_scheduler
    assert cfg.herpn_require_full_conversion
    assert sum(map(len, cfg.herpn_conversion_groups)) == 25
    final_conversion_epoch = (
        cfg.herpn_group_epochs[-1] + cfg.herpn_transition_epochs)
    assert final_conversion_epoch == 20.0
    assert cfg.num_epoch == 24

    assert cfg.herpn_range_loss_weight == 0.0
    assert cfg.herpn_distill_loss_weight == 0.0
    assert cfg.herpn_bn_recalibration_batches == 0
    assert cfg.max_nonfinite_embedding_skips > 0
    assert cfg.max_nonfinite_loss_skips > 0
    assert cfg.skip_nonfinite_gradients
    assert cfg.max_nonfinite_gradient_skips > 0
    assert not cfg.fail_on_nonfinite_val
    assert cfg.max_validation_embedding_abs is None


def test_layerwise_scaled_r50_config_schedules_forward_order_and_augmentation():
    cfg = _load_standalone_config(
        "configs/ms1mv3_r50_prelu_herpn_layerwise_scale.py")
    model = get_model(
        cfg.network,
        dropout=0,
        fp16=False,
        herpn_range_limit=cfg.herpn_range_limit,
        herpn_bn_eps=cfg.herpn_bn_eps,
        herpn_progress=cfg.herpn_initial_progress,
        prelu_herpn_distill_eps=cfg.prelu_herpn_distill_eps,
        prelu_herpn_layerwise_scale=cfg.prelu_herpn_layerwise_scale,
        prelu_herpn_initial_scale=cfg.prelu_herpn_initial_scale,
    )
    model_order = tuple(model.layerwise_poly_activation_names())
    configured_order = tuple(
        name for group in cfg.herpn_conversion_groups for name in group)

    assert model.layerwise_input_scale_enabled
    assert len(model_order) == 25
    assert configured_order == model_order
    assert all(len(group) == 1 for group in cfg.herpn_conversion_groups)
    assert cfg.herpn_range_limit == 1.0
    assert cfg.layerwise_poly_range_margin == 2.0
    assert cfg.range_augmentation["enabled"]
    assert cfg.range_augmentation["probability"] >= 0.5
    assert cfg.herpn_group_epochs == tuple(
        index + 0.5 for index in range(25))
    assert cfg.herpn_transition_epochs == 0.5
    assert all(
        right - left == 1.0
        for left, right in zip(
            cfg.herpn_group_epochs, cfg.herpn_group_epochs[1:]))
    final_conversion_epoch = (
        cfg.herpn_group_epochs[-1] + cfg.herpn_transition_epochs)
    assert final_conversion_epoch == 25.0
    assert cfg.num_epoch - final_conversion_epoch == 4.0
    assert cfg.output.endswith("layerwise_scale_range_aug_one_epoch")


def test_nl9_fixed_recovery_keeps_all_nine_quadratics_from_epoch_zero():
    clean = _load_standalone_config(
        "configs/ms1mv3_r50_nl9_prelu_herpn_fixed_recovery.py")
    augmented = _load_standalone_config(
        "configs/ms1mv3_r50_nl9_prelu_herpn_fixed_recovery_aug.py")
    scheduled = tuple(
        name for group in clean.herpn_conversion_groups for name in group)

    assert scheduled == NL9_ACTIVATION_NAMES
    assert all(
        start + clean.herpn_transition_epochs <= 0.0
        for start in clean.herpn_group_epochs)
    assert clean.herpn_require_full_conversion
    assert clean.embedding_teacher_network == "r50_nl9"
    assert clean.embedding_distill_weight == pytest.approx(1.0)
    assert clean.fail_on_nonfinite_val
    assert augmented.embedding_distill_weight == pytest.approx(5.0)
    assert augmented.range_augmentation["enabled"]
    assert augmented.range_augmentation["probability"] == pytest.approx(0.5)

    bnfreeze = _load_standalone_config(
        "configs/ms1mv3_r50_nl9_prelu_herpn_fixed_recovery_bnfreeze.py")
    assert bnfreeze.lr == pytest.approx(5.0e-5)
    assert bnfreeze.freeze_batchnorm_running_stats
    assert bnfreeze.freeze_batchnorm_affine
    assert bnfreeze.herpn_range_loss_weight == 0.0
    assert bnfreeze.herpn_distill_loss_weight == 0.0
    assert bnfreeze.max_nonfinite_embedding_skips == 1000


def test_layerwise_scaled_r50_group4_recovery_conditions_layer2_before_blend():
    cfg = _load_standalone_config(
        "configs/"
        "ms1mv3_r50_prelu_herpn_layerwise_scale_recover_group4.py")

    assert not cfg.resume
    assert cfg.backbone_init.endswith(
        "model_herpn_group_04_bnrecalibrated.pt")
    assert cfg.output.endswith("range_aug_recover_group4")
    assert cfg.backbone_init_herpn_progress == pytest.approx(2.0)

    assert cfg.layerwise_poly_initial_calibration_provisional
    assert cfg.layerwise_poly_staged_training
    assert not cfg.layerwise_poly_freeze_backbone_during_local_fit
    assert cfg.layerwise_poly_allow_provisional_tail_conditioning
    assert cfg.layerwise_poly_conditioning_backbone_lr_scale == pytest.approx(
        0.01)
    assert cfg.layerwise_poly_conditioning_range_loss_weight == pytest.approx(
        1.0)
    assert cfg.layerwise_poly_strict_recalibrate_before_blend
    assert cfg.layerwise_poly_strict_tail_scale_floor
    assert cfg.layerwise_poly_tail_scale_floor_margin == pytest.approx(1.1)
    assert cfg.layerwise_poly_max_tail_scale_expansion == pytest.approx(2.0)

    starts = cfg.herpn_group_epochs
    transition = cfg.herpn_transition_epochs
    assert len(starts) == len(cfg.herpn_conversion_groups) == 25
    assert all(start + transition <= 0.0 for start in starts[:4])
    assert starts[4] == pytest.approx(1.0)
    assert starts[5] == pytest.approx(2.5)
    assert cfg.herpn_conversion_groups[4] == (("layer2.0.prelu",))
    assert all(
        right >= left + transition
        for left, right in zip(starts, starts[1:]))
    final_conversion_epoch = starts[-1] + transition
    assert final_conversion_epoch == pytest.approx(22.0)
    assert cfg.num_epoch - final_conversion_epoch == pytest.approx(4.0)


def test_selective_weight_decay_protects_herpn_norm_and_bias():
    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 4, 3, bias=True)
            self.bn = nn.BatchNorm2d(4)
            self.activation = PReLUHerPNActivation(4, blend=0.0)
            self.linear = nn.Linear(4, 2, bias=False)

    model = TinyModel()
    decay, no_decay = split_weight_decay_parameters(model)
    decay_ids = {id(parameter) for parameter in decay}
    no_decay_ids = {id(parameter) for parameter in no_decay}

    assert id(model.conv.weight) in decay_ids
    assert id(model.linear.weight) in decay_ids
    assert id(model.conv.bias) in no_decay_ids
    assert id(model.bn.weight) in no_decay_ids
    assert id(model.bn.bias) in no_decay_ids
    assert id(model.activation.herpn.weight) in no_decay_ids
    assert id(model.activation.herpn.bias) in no_decay_ids
    assert id(model.activation.prelu.weight) not in decay_ids | no_decay_ids
    assert decay_ids.isdisjoint(no_decay_ids)

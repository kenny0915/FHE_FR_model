import importlib.util
import sys
import types
from pathlib import Path

import torch

from backbones import get_model
from backbones.iresnet_pillar import PILLARPolynomialReLU
from lr_scheduler import CosineLRWarmup
from utils.utils_pillar import (
    pillar_regularization_at_epoch,
    pillar_task_loss_weight_at_epoch,
)


class _EasyDict(dict):
    __setattr__ = dict.__setitem__

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as error:
            raise AttributeError(key) from error


def _load_standalone_config(filename):
    fake_easydict = types.ModuleType("easydict")
    fake_easydict.EasyDict = _EasyDict
    previous = sys.modules.get("easydict")
    sys.modules["easydict"] = fake_easydict
    try:
        path = Path(__file__).parents[1] / "configs" / filename
        spec = importlib.util.spec_from_file_location(
            f"_test_{path.stem}", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.config
    finally:
        if previous is None:
            del sys.modules["easydict"]
        else:
            sys.modules["easydict"] = previous


def _paper_polynomial(x):
    square = x.square()
    return (
        322.0 / 1024.0
        + 0.5 * x
        + 160.0 / 1024.0 * square
        - 3.0 / 1024.0 * square.square()
    )


def test_pillar_eval_is_the_unclipped_paper_polynomial():
    activation = PILLARPolynomialReLU().eval()
    activation.set_range_tracking(True)
    inputs = torch.tensor([-6.0, -5.0, 0.0, 5.0, 6.0])

    actual = activation(inputs)

    torch.testing.assert_close(actual, _paper_polynomial(inputs))
    assert not torch.isclose(actual[-1], _paper_polynomial(inputs[-1].clamp(-5, 5)))
    assert activation.polynomial_degree == 4
    assert activation.multiplicative_depth == 2
    assert list(activation.parameters()) == []
    assert activation.range_penalty() is None
    assert activation.range_stats()["absmax"] == 6.0

    inference_graph = str(torch.jit.trace(
        PILLARPolynomialReLU().eval(), inputs).inlined_graph)
    assert "aten::clamp" not in inference_graph
    assert "aten::relu" not in inference_graph
    assert "aten::prelu" not in inference_graph


def test_pillar_training_penalizes_raw_input_before_clipping():
    activation = PILLARPolynomialReLU(
        approximation_range=5.0,
        regularization_range=4.8,
        regularization_exponent=4,
        training_clip=True,
    ).train()
    inputs = torch.tensor([-4.8, 9.6], requires_grad=True)

    actual = activation(inputs)
    penalty = activation.range_penalty()

    torch.testing.assert_close(actual, _paper_polynomial(inputs.clamp(-5, 5)))
    torch.testing.assert_close(penalty, torch.tensor((1.0 + 16.0) / 2.0))
    stats = activation.range_stats()
    torch.testing.assert_close(
        stats["approximation_outside_fraction"], torch.tensor(0.5))
    torch.testing.assert_close(
        stats["regularization_outside_fraction"], torch.tensor(0.5))
    penalty.backward()
    assert inputs.grad is not None
    assert torch.isfinite(inputs.grad).all()


def test_pillar_released_sum_penalty_is_unnormalized():
    activation = PILLARPolynomialReLU(
        regularization_range=4.8,
        regularization_exponent=4,
        penalty_reduction="sum",
    ).train()
    inputs = torch.tensor([-4.8, 9.6], requires_grad=True)

    activation(inputs)

    torch.testing.assert_close(
        activation.range_penalty(), torch.tensor(1.0 + 16.0))


def test_pillar_range_only_task_schedule_matches_released_code():
    assert pillar_task_loss_weight_at_epoch(0, range_only_epochs=1) == 0.0
    assert pillar_task_loss_weight_at_epoch(1, range_only_epochs=1) == 1.0
    assert pillar_task_loss_weight_at_epoch(0, range_only_epochs=0) == 1.0


def test_pillar_warmup_matches_paper_schedule():
    expected = (
        (5e-7, 4),
        (1e-6, 6),
        (5e-6, 8),
        (1e-5, 10),
        (5e-5, 10),
    )
    actual = tuple(
        pillar_regularization_at_epoch(epoch, 5e-5, 10)
        for epoch in range(5)
    )
    for (actual_beta, actual_gamma), (expected_beta, expected_gamma) in zip(
            actual, expected):
        assert actual_beta == expected_beta
        assert actual_gamma == expected_gamma


def test_cosine_scheduler_has_linear_warmup_and_minimum_ratio():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    scheduler = CosineLRWarmup(
        optimizer, warmup_iters=5, total_iters=15, min_lr_ratio=0.01)

    assert scheduler._scale(0) == 0.0
    assert scheduler._scale(5) == 1.0
    assert scheduler._scale(10) == 0.505
    assert scheduler._scale(15) == 0.01


def test_r50_pillar_backbone_and_configs_cover_the_recipe():
    cfg = _load_standalone_config("ms1mv3_r50_pillar.py")
    model = get_model(
        cfg.network,
        dropout=0,
        fp16=False,
        pillar_approximation_range=cfg.pillar_approximation_range,
        pillar_regularization_range=cfg.pillar_regularization_range,
        pillar_regularization_exponent=cfg.pillar_regularization_exponent,
        pillar_training_clip=cfg.pillar_training_clip,
    )
    activations = model.pillar_activations()

    assert len(activations) == 25
    assert all(
        isinstance(activation, PILLARPolynomialReLU)
        for activation in activations)
    assert not any(isinstance(module, torch.nn.PReLU)
                   for module in model.modules())
    assert cfg.pillar_approximation_range == 5.0
    assert cfg.pillar_regularization_range == 4.8
    assert cfg.pillar_regularization_coefficient == 5e-5
    assert cfg.pillar_regularization_exponent == 10
    assert cfg.lr_scheduler == "cosine"
    assert cfg.fp16 is False

    smoke = _load_standalone_config("casia_r50_pillar_smoke.py")
    assert smoke.network == "r50_pillar"
    assert smoke.max_steps_per_epoch == 100
    assert smoke.pillar_regularization_coefficient == 5e-5
    assert smoke.val_targets == []

    released = _load_standalone_config("ms1mv3_r50_pillar_espn.py")
    assert released.pillar_penalty_reduction == "sum"
    assert released.pillar_range_only_epochs == 1
    assert released.pillar_regularization_coefficient == 1e-4
    assert released.batch_size == 256
    assert released.weight_decay == 2e-5
    assert released.selective_weight_decay is True
    assert released.num_epoch == 32


def test_r18_pillar_lightweight_forward_and_layer_averaged_penalty():
    torch.manual_seed(7)
    model = get_model(
        "r18_pillar", dropout=0, fp16=False,
        pillar_regularization_exponent=4,
    ).train()
    inputs = torch.randn(2, 3, 112, 112)

    outputs = model(inputs)
    penalty = model.pillar_range_penalty()

    assert outputs.shape == (2, 512)
    assert len(model.pillar_activations()) == 9
    assert penalty.ndim == 0
    assert torch.isfinite(outputs).all()
    assert torch.isfinite(penalty)

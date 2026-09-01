import torch

from backbones.iresnet_no_relu import IResNet, IBasicBlock, ProgressiveHerPNActivation
from probe_herpn_quadratic_scales import (
    attenuate_quadratic_basis_variance,
)


def test_linear_tail_penalty_focuses_extreme_values_and_stays_finite():
    activation = ProgressiveHerPNActivation(
        2,
        range_limit=6.0,
        range_penalty_mode="linear_tail",
        range_topk_fraction=0.25,
        range_bulk_weight=0.01,
        training_stabilization_limit=6.0,
    ).train()
    inputs = torch.tensor(
        [[[[0.0, 7.0], [8.0, 1.0]], [[2.0, 3.0], [4.0, 60.0]]]],
        requires_grad=True,
    )

    outputs = activation(inputs)
    penalty = activation.range_penalty()
    (outputs.square().mean() + penalty).backward()

    assert torch.isfinite(outputs).all()
    assert torch.isfinite(penalty)
    assert penalty.item() > 1.0
    assert torch.isfinite(inputs.grad).all()
    assert inputs.grad[0, 1, 1, 1].abs().item() > 0.0


def test_training_surrogate_does_not_change_eval_forward():
    torch.manual_seed(7)
    bounded = ProgressiveHerPNActivation(
        2, blend=1.0, training_stabilization_limit=6.0)
    exact = ProgressiveHerPNActivation(
        2, blend=1.0, training_stabilization_limit=None)
    exact.load_state_dict(bounded.state_dict(), strict=True)
    bounded.eval()
    exact.eval()
    inputs = torch.randn(2, 2, 3, 3) * 20.0

    torch.testing.assert_close(bounded(inputs), exact(inputs))


def test_named_range_penalty_selects_only_requested_activation():
    model = IResNet(
        IBasicBlock,
        [1, 1, 1, 1],
        herpn_progress=5.0,
        herpn_range_penalty_mode="linear_tail",
    ).train()
    model(torch.randn(2, 3, 112, 112))

    selected = model.herpn_range_penalty(("layer3.0.prelu",))
    expected = model.layer3[0].prelu.range_penalty()
    torch.testing.assert_close(selected, expected)

    try:
        model.herpn_range_penalty(("layer3.99.prelu",))
    except ValueError as error:
        assert "Unknown HerPN activation" in str(error)
    else:
        raise AssertionError("unknown activation name was silently accepted")


def test_training_stabilization_can_start_at_a_named_boundary():
    model = IResNet(
        IBasicBlock,
        [1, 1, 1, 1],
        herpn_progress=5.0,
        herpn_training_stabilization_limit=6.0,
        herpn_training_stabilization_names=(
            "layer2.0.prelu", "layer3.0.prelu", "layer4.0.prelu"),
    )
    activations = {
        name: module for name, module in model.named_modules()
        if isinstance(module, ProgressiveHerPNActivation)
    }
    assert activations["layer1.0.prelu"].training_stabilization_limit is None
    assert activations["layer2.0.prelu"].training_stabilization_limit == 6.0
    assert activations["layer3.0.prelu"].training_stabilization_limit == 6.0


def test_quadratic_basis_variance_attenuates_folded_coefficient():
    activation = ProgressiveHerPNActivation(3, bn_eps=1e-4, blend=1.0)
    activation.eval()
    original = activation.herpn.folded_coefficients()[0]
    activation.herpn.bn2.running_var.copy_(
        attenuate_quadratic_basis_variance(
            activation.herpn.bn2.running_var,
            activation.herpn.bn2.eps,
            0.125,
        )
    )
    adjusted = activation.herpn.folded_coefficients()[0]
    torch.testing.assert_close(adjusted, original * 0.125)

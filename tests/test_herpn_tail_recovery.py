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


def test_independent_basis_scales_load_legacy_herpn_exactly():
    torch.manual_seed(13)
    legacy = ProgressiveHerPNActivation(
        4, bn_eps=1e-4, blend=1.0).eval()
    with torch.no_grad():
        legacy.herpn.weight.uniform_(0.7, 1.3)
        legacy.herpn.bias.uniform_(-0.2, 0.2)
        for batchnorm in (
                legacy.herpn.bn0, legacy.herpn.bn1, legacy.herpn.bn2):
            batchnorm.running_mean.uniform_(-0.5, 0.5)
            batchnorm.running_var.uniform_(0.4, 1.8)

    independent = ProgressiveHerPNActivation(
        4,
        bn_eps=1e-4,
        blend=1.0,
        independent_basis_scales=True,
    ).eval()
    independent.load_state_dict(legacy.state_dict(), strict=True)

    inputs = torch.randn(3, 4, 5, 5)
    torch.testing.assert_close(
        independent(inputs), legacy(inputs), rtol=0.0, atol=0.0)
    for actual, expected in zip(
            independent.herpn.folded_coefficients(),
            legacy.herpn.folded_coefficients()):
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        independent.herpn.basis_scale,
        torch.ones_like(independent.herpn.basis_scale),
    )


def test_independent_basis_scales_remain_exact_foldable_quadratic():
    torch.manual_seed(17)
    activation = ProgressiveHerPNActivation(
        3,
        bn_eps=1e-4,
        blend=1.0,
        independent_basis_scales=True,
    ).eval()
    with torch.no_grad():
        activation.herpn.basis_scale[0].fill_(0.8)
        activation.herpn.basis_scale[1].fill_(1.1)
        activation.herpn.basis_scale[2].fill_(0.65)
    inputs = torch.randn(2, 3, 4, 4, requires_grad=True)
    exact = activation(inputs)
    folded = activation.folded()(inputs)
    torch.testing.assert_close(folded, exact, rtol=1e-5, atol=1e-6)

    exact.square().mean().backward()
    scale_gradient = activation.herpn.basis_scale.grad
    assert scale_gradient is not None
    assert torch.isfinite(scale_gradient).all()
    assert (scale_gradient.flatten(1).abs().sum(dim=1) > 0).all()


def test_basis_anchor_covers_all_independent_herpn_activations():
    model = IResNet(
        IBasicBlock,
        [1, 1, 1, 1],
        herpn_progress=5.0,
        herpn_independent_basis_scales=True,
    )
    assert model.herpn_basis_anchor_loss().item() == 0.0
    with torch.no_grad():
        model.layer2[0].prelu.herpn.basis_scale[2].fill_(0.5)
    assert model.herpn_basis_anchor_loss().item() > 0.0

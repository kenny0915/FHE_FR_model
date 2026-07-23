import torch

from eval_poolformer_checkpoint_fp32 import (
    extract_backbone_state,
    gate_blends_for_epoch,
    infer_completed_epoch,
)


def test_infers_epoch_from_epoch_snapshot_name():
    assert infer_completed_epoch("/tmp/model_epoch_10.pt") == 10
    assert infer_completed_epoch("/tmp/model.pt") is None


def test_reconstructs_epoch_10_progressive_gate_blends():
    blends = gate_blends_for_epoch(
        completed_epoch=10,
        group_epochs=(8, 10, 12, 14, 16, 18),
        transition_epochs=2.0,
    )
    assert blends == (1.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def test_extracts_full_training_checkpoint_and_removes_ddp_prefix():
    checkpoint = {
        "epoch": 10,
        "state_dict_backbone": {
            "module.weight": torch.ones(1),
            "module.bias": torch.zeros(1),
        },
    }
    state = extract_backbone_state(checkpoint)
    assert set(state) == {"weight", "bias"}

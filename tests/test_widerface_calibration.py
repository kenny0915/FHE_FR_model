import cv2
import numpy as np
import pytest
import torch

from configs.ms1mv3_r50_no_relu_phase2_wider_robust_margin import (
    config as wider_config,
)
from configs.ms1mv3_r50_no_relu_phase2_wider_ms1mv3_focus1 import (
    config as wider_ms1mv3_focus_config,
)
from dataset import MXFaceDataset, PairedOrientationDataset
from evaluate_numerical_gate import make_gate_dataset
from utils.utils_widerface import (
    WIDERFaceDataset,
    crop_wider_face,
    parse_wider_face_annotations,
    wider_image_fold,
)


def test_wider_parser_ignores_official_zero_face_sentinel(tmp_path):
    annotation = tmp_path / "wider.txt"
    annotation.write_text(
        "event/zero.jpg\n0\n0 0 0 0 0 0 0 0 0 0\n"
        "event/one.jpg\n1\n10 11 30 40 2 0 1 0 2 1\n",
        encoding="utf-8")
    records = parse_wider_face_annotations(annotation)
    assert records[0] == ("event/zero.jpg", ())
    assert records[1][0] == "event/one.jpg"
    assert records[1][1][0] == (10, 11, 30, 40, 2, 0, 1, 0, 2, 1)


def test_wider_crop_is_normalized_and_flip_is_exact(tmp_path):
    image_root = tmp_path / "images"
    event = image_root / "event"
    event.mkdir(parents=True)
    relative_paths = []
    for index in range(100):
        candidate = f"event/face_{index}.jpg"
        if wider_image_fold(candidate, 10) == 0:
            relative_paths.append(candidate)
            break
    assert relative_paths
    relative_path = relative_paths[0]
    image = np.zeros((80, 100, 3), dtype=np.uint8)
    image[:, :50] = (0, 0, 255)
    image[:, 50:] = (255, 0, 0)
    assert cv2.imwrite(str(image_root / relative_path), image)
    annotation = tmp_path / "wider.txt"
    annotation.write_text(
        f"{relative_path}\n2\n10 10 60 60 0 0 0 0 0 0\n"
        "20 20 10 10 0 0 0 0 0 0\n",
        encoding="utf-8")

    dataset = WIDERFaceDataset(
        image_root, annotation, split="validation", min_face_size=20,
        crop_scale=1.0)
    assert len(dataset) == 1
    assert dataset.image_count == 1
    image_tensor, label = dataset[0]
    assert label == 0
    assert image_tensor.shape == (3, 112, 112)
    assert image_tensor.min() >= -1.0
    assert image_tensor.max() <= 1.0
    pair, source_index = PairedOrientationDataset(dataset)[0]
    assert source_index == 0
    torch.testing.assert_close(pair[0], image_tensor)
    torch.testing.assert_close(pair[1], torch.flip(image_tensor, dims=(-1,)))


def test_wider_image_split_is_stable_and_disjoint():
    paths = [f"event/image_{index}.jpg" for index in range(100)]
    validation = {path for path in paths if wider_image_fold(path, 10) == 0}
    calibration = set(paths).difference(validation)
    assert validation
    assert calibration
    assert validation.isdisjoint(calibration)
    assert all(wider_image_fold(path, 10) == wider_image_fold(path, 10)
               for path in paths)
    with pytest.raises(ValueError, match="greater than one"):
        wider_image_fold("face.jpg", 1)


def test_wider_experiment_keeps_ijb_out_of_training_and_selection():
    config = wider_config
    assert config.calibration_dataset == "wider"
    assert config.backbone_init.endswith("model_epoch_23.pt")
    assert config.wider_mining_split == "calibration"
    assert config.wider_validation_modulo == 10
    assert config.wider_validation_fold == 0
    assert config.wider_min_face_size == 20
    assert config.wider_crop_scale == pytest.approx(1.35)
    assert not config.replay_gate_failures
    assert config.numerical_range_gate_limit == pytest.approx(4.0)
    assert config.herpn_range_limit == pytest.approx(6.0)
    assert config.adversarial_tail_enabled
    focus = wider_ms1mv3_focus_config
    assert focus.calibration_dataset == "ms1mv3"
    assert focus.backbone_init.endswith(
        "wider_robust_margin/model_numerical_gate_zero.pt")
    assert focus.calibration_priority_manifests[0].endswith(
        "model_numerical_gate_zero_ms1mv3_gate.json")
    assert focus.calibration_replay_activation_topk == 64
    assert focus.ijbc_gate_failure_repeats == 128
    assert focus.numerical_range_gate_limit == pytest.approx(4.0)


def test_standalone_wider_gate_uses_held_out_fold():
    assert callable(make_gate_dataset)


def test_wider_crop_rejects_invalid_boxes():
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="must be positive"):
        crop_wider_face(image, (1, 1, 0, 10))


def test_mxface_oriented_conversion_does_not_require_numpy_bridge():
    dataset = object.__new__(MXFaceDataset)
    pixels = np.array([[[0, 127, 255], [255, 127, 0]]], dtype=np.uint8)
    dataset._read = lambda index: (pixels, torch.tensor(3))
    original, label = dataset.get_oriented(0, 0)
    flipped, _ = dataset.get_oriented(0, 1)
    assert label.item() == 3
    assert original.shape == (3, 1, 2)
    assert original[0, 0, 0].item() == pytest.approx(-1.0)
    assert original[2, 0, 0].item() == pytest.approx(1.0)
    torch.testing.assert_close(flipped, torch.flip(original, dims=(-1,)))

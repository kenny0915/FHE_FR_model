import torch
from torch.utils.data import Dataset

from dataset import DatasetWithIndex


class _TinyDataset(Dataset):
    def __len__(self):
        return 2

    def __getitem__(self, index):
        image = torch.tensor([[index * 10 + 1, index * 10 + 2]])
        return image, torch.tensor(index + 7)


def test_dataset_with_index_can_scan_both_orientations():
    wrapped = DatasetWithIndex(_TinyDataset(), both_orientations=True)

    assert len(wrapped) == 4
    first, first_label, first_index, first_orientation = wrapped[0]
    flipped, flipped_label, flipped_index, flipped_orientation = wrapped[1]
    second, _, second_index, _ = wrapped[2]

    assert torch.equal(first, torch.tensor([[1, 2]]))
    assert torch.equal(flipped, torch.tensor([[2, 1]]))
    assert first_label == flipped_label == 7
    assert (first_index, flipped_index, second_index) == (0, 0, 1)
    assert (first_orientation, flipped_orientation) == (0, 1)
    assert torch.equal(second, torch.tensor([[11, 12]]))


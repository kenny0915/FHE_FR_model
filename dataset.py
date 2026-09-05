import numbers
import os
import queue as Queue
import threading
from typing import Iterable

import numpy as np

if not hasattr(np, 'PZERO'):
    np.PZERO = 0.0
if not hasattr(np, 'NZERO'):
    np.NZERO = -0.0
if not hasattr(np, 'PINF'):
    np.PINF = np.inf
if not hasattr(np, 'Inf'):
    np.Inf = np.inf
if not hasattr(np, 'NINF'):
    np.NINF = -np.inf
if not hasattr(np, 'NAN'):
    np.NAN = np.nan
if not hasattr(np, 'NaN'):
    np.NaN = np.nan
if not hasattr(np, 'alltrue'):
    np.alltrue = np.all
if not hasattr(np, 'sometrue'):
    np.sometrue = np.any
def _mxnet_unavailable_financial(*args, **kwargs):
    raise NotImplementedError('NumPy financial functions are not available in this environment.')
if not hasattr(np, 'msort'):
    np.msort = lambda a: np.sort(a, axis=0)
if not hasattr(np, 'round_'):
    np.round_ = np.round
if not hasattr(np, 'product'):
    np.product = np.prod
for _name in ('fv', 'pmt', 'nper', 'ipmt', 'ppmt', 'pv', 'rate', 'irr', 'npv', 'mirr'):
    if not hasattr(np, _name):
        setattr(np, _name, _mxnet_unavailable_financial)
import mxnet as mx
import torch
from functools import partial
from torch import distributed
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from utils.utils_distributed_sampler import DistributedSampler
from utils.utils_distributed_sampler import get_dist_info, worker_init_fn
from utils.utils_range_augmentation import (
    RangeStressAugmentation,
    range_augmentation_enabled,
)


def _face_training_transform(range_augmentation=None, to_pil=False):
    operations = []
    if to_pil:
        operations.append(transforms.ToPILImage())
    operations.extend((
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ))
    if range_augmentation_enabled(range_augmentation):
        operations.append(RangeStressAugmentation(range_augmentation))
    operations.append(
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
    return transforms.Compose(operations)


def get_dataloader(
    root_dir,
    local_rank,
    batch_size,
    dali = False,
    dali_aug = False,
    seed = 2048,
    num_workers = 2,
    drop_last = True,
    range_augmentation = None,
    ) -> Iterable:

    rec = os.path.join(root_dir, 'train.rec')
    idx = os.path.join(root_dir, 'train.idx')
    train_set = None

    # Synthetic
    if root_dir == "synthetic":
        train_set = SyntheticDataset()
        dali = False

    # Mxnet RecordIO
    elif os.path.exists(rec) and os.path.exists(idx):
        train_set = MXFaceDataset(
            root_dir=root_dir,
            local_rank=local_rank,
            range_augmentation=range_augmentation)

    # Image Folder
    else:
        transform = _face_training_transform(range_augmentation)
        train_set = ImageFolder(root_dir, transform)

    # DALI
    if dali:
        if range_augmentation_enabled(range_augmentation):
            raise ValueError(
                "range_augmentation currently requires config.dali=False")
        return dali_data_iter(
            batch_size=batch_size, rec_file=rec, idx_file=idx,
            num_threads=2, local_rank=local_rank, dali_aug=dali_aug)

    rank, world_size = get_dist_info()
    train_sampler = DistributedSampler(
        train_set, num_replicas=world_size, rank=rank, shuffle=True, seed=seed)

    if seed is None:
        init_fn = None
    else:
        init_fn = partial(worker_init_fn, num_workers=num_workers, rank=rank, seed=seed)

    train_loader = DataLoaderX(
        local_rank=local_rank,
        dataset=train_set,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
        worker_init_fn=init_fn,
    )

    return train_loader

class BackgroundGenerator(threading.Thread):
    def __init__(self, generator, local_rank, max_prefetch=6):
        super(BackgroundGenerator, self).__init__()
        self.queue = Queue.Queue(max_prefetch)
        self.generator = generator
        self.local_rank = local_rank
        self.daemon = True
        self.start()

    def run(self):
        torch.cuda.set_device(self.local_rank)
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def next(self):
        next_item = self.queue.get()
        if next_item is None:
            raise StopIteration
        return next_item

    def __next__(self):
        return self.next()

    def __iter__(self):
        return self


class DataLoaderX(DataLoader):

    def __init__(self, local_rank, **kwargs):
        super(DataLoaderX, self).__init__(**kwargs)
        self.stream = torch.cuda.Stream(local_rank)
        self.local_rank = local_rank

    def __iter__(self):
        self.iter = super(DataLoaderX, self).__iter__()
        self.iter = BackgroundGenerator(self.iter, self.local_rank)
        self.preload()
        return self

    def preload(self):
        self.batch = next(self.iter, None)
        if self.batch is None:
            return None
        with torch.cuda.stream(self.stream):
            for k in range(len(self.batch)):
                self.batch[k] = self.batch[k].to(device=self.local_rank, non_blocking=True)

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        if batch is None:
            raise StopIteration
        self.preload()
        return batch


class MXFaceDataset(Dataset):
    def __init__(self, root_dir, local_rank, range_augmentation=None):
        super(MXFaceDataset, self).__init__()
        self.transform = _face_training_transform(
            range_augmentation, to_pil=True)
        self.oriented_transform = transforms.Compose((
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ))
        self.root_dir = root_dir
        self.local_rank = local_rank
        path_imgrec = os.path.join(root_dir, 'train.rec')
        path_imgidx = os.path.join(root_dir, 'train.idx')
        self.imgrec = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')
        s = self.imgrec.read_idx(0)
        header, _ = mx.recordio.unpack(s)
        if header.flag > 0:
            self.header0 = (int(header.label[0]), int(header.label[1]))
            self.imgidx = np.array(range(1, int(header.label[0])))
        else:
            self.imgidx = np.array(list(self.imgrec.keys))

    def _read(self, index):
        idx = self.imgidx[index]
        s = self.imgrec.read_idx(idx)
        header, img = mx.recordio.unpack(s)
        label = header.label
        if not isinstance(label, numbers.Number):
            label = label[0]
        label = torch.tensor(int(label), dtype=torch.long)
        sample = mx.image.imdecode(img).asnumpy()
        return sample, label

    def __getitem__(self, index):
        sample, label = self._read(index)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, label

    def get_oriented(self, index, orientation):
        """Return a deterministic canonical or horizontally flipped image."""
        sample, label = self._read(index)
        sample = self.oriented_transform(sample)
        if int(orientation):
            sample = torch.flip(sample, dims=(-1,))
        return sample, label

    def __len__(self):
        return len(self.imgidx)


class DatasetWithIndex(Dataset):
    """Expose the stable source index without changing the training dataset.

    Range calibration uses this wrapper to persist the identities of rare-tail
    samples. Ordinary training keeps the original two-item ``(image, label)``
    interface, so existing callers and DALI remain unchanged.
    """

    def __init__(self, dataset, both_orientations=False):
        super().__init__()
        self.dataset = dataset
        self.both_orientations = bool(both_orientations)

    def __getitem__(self, index):
        orientation = int(index % 2) if self.both_orientations else 0
        source_index = int(index // 2) if self.both_orientations else int(index)
        oriented_getter = getattr(self.dataset, "get_oriented", None)
        sample = (
            oriented_getter(source_index, orientation)
            if self.both_orientations and oriented_getter is not None
            else self.dataset[source_index]
        )
        if not isinstance(sample, (tuple, list)) or len(sample) != 2:
            raise ValueError(
                "DatasetWithIndex expects (image, label) dataset samples")
        image = sample[0]
        if orientation and not (
                self.both_orientations and oriented_getter is not None):
            image = torch.flip(image, dims=(-1,))
        return image, sample[1], source_index, orientation

    def __len__(self):
        multiplier = 2 if self.both_orientations else 1
        return multiplier * len(self.dataset)


class PairedOrientationDataset(Dataset):
    """Return deterministic original/flip pairs for a training dataset."""

    def __init__(self, dataset):
        super().__init__()
        if not callable(getattr(dataset, "get_oriented", None)):
            raise ValueError(
                "PairedOrientationDataset requires dataset.get_oriented")
        self.dataset = dataset

    def __getitem__(self, index):
        index = int(index)
        pair_getter = getattr(self.dataset, "get_pair", None)
        if callable(pair_getter):
            pair = pair_getter(index)
        else:
            original = self.dataset.get_oriented(index, 0)[0]
            flipped = self.dataset.get_oriented(index, 1)[0]
            pair = torch.stack((original, flipped))
        return pair, index

    def __len__(self):
        return len(self.dataset)


class SyntheticDataset(Dataset):
    def __init__(self):
        super(SyntheticDataset, self).__init__()
        img = np.random.randint(0, 255, size=(112, 112, 3), dtype=np.int32)
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).squeeze(0).float()
        img = ((img / 255) - 0.5) / 0.5
        self.img = img
        self.label = 1

    def __getitem__(self, index):
        return self.img, self.label

    def __len__(self):
        return 1000000


def dali_data_iter(
    batch_size: int, rec_file: str, idx_file: str, num_threads: int,
    initial_fill=32768, random_shuffle=True,
    prefetch_queue_depth=1, local_rank=0, name="reader",
    mean=(127.5, 127.5, 127.5), 
    std=(127.5, 127.5, 127.5),
    dali_aug=False
    ):
    """
    Parameters:
    ----------
    initial_fill: int
        Size of the buffer that is used for shuffling. If random_shuffle is False, this parameter is ignored.

    """
    rank: int = distributed.get_rank()
    world_size: int = distributed.get_world_size()
    import nvidia.dali.fn as fn
    import nvidia.dali.types as types
    from nvidia.dali.pipeline import Pipeline
    from nvidia.dali.plugin.pytorch import DALIClassificationIterator

    def dali_random_resize(img, resize_size, image_size=112):
        img = fn.resize(img, resize_x=resize_size, resize_y=resize_size)
        img = fn.resize(img, size=(image_size, image_size))
        return img
    def dali_random_gaussian_blur(img, window_size):
        img = fn.gaussian_blur(img, window_size=window_size * 2 + 1)
        return img
    def dali_random_gray(img, prob_gray):
        saturate = fn.random.coin_flip(probability=1 - prob_gray)
        saturate = fn.cast(saturate, dtype=types.FLOAT)
        img = fn.hsv(img, saturation=saturate)
        return img
    def dali_random_hsv(img, hue, saturation):
        img = fn.hsv(img, hue=hue, saturation=saturation)
        return img
    def multiplexing(condition, true_case, false_case):
        neg_condition = condition ^ True
        return condition * true_case + neg_condition * false_case

    condition_resize = fn.random.coin_flip(probability=0.1)
    size_resize = fn.random.uniform(range=(int(112 * 0.5), int(112 * 0.8)), dtype=types.FLOAT)
    condition_blur = fn.random.coin_flip(probability=0.2)
    window_size_blur = fn.random.uniform(range=(1, 2), dtype=types.INT32)
    condition_flip = fn.random.coin_flip(probability=0.5)
    condition_hsv = fn.random.coin_flip(probability=0.2)
    hsv_hue = fn.random.uniform(range=(0., 20.), dtype=types.FLOAT)
    hsv_saturation = fn.random.uniform(range=(1., 1.2), dtype=types.FLOAT)

    pipe = Pipeline(
        batch_size=batch_size, num_threads=num_threads,
        device_id=local_rank, prefetch_queue_depth=prefetch_queue_depth, )
    condition_flip = fn.random.coin_flip(probability=0.5)
    with pipe:
        jpegs, labels = fn.readers.mxnet(
            path=rec_file, index_path=idx_file, initial_fill=initial_fill, 
            num_shards=world_size, shard_id=rank,
            random_shuffle=random_shuffle, pad_last_batch=False, name=name)
        images = fn.decoders.image(jpegs, device="mixed", output_type=types.RGB)
        if dali_aug:
            images = fn.cast(images, dtype=types.UINT8)
            images = multiplexing(condition_resize, dali_random_resize(images, size_resize, image_size=112), images)
            images = multiplexing(condition_blur, dali_random_gaussian_blur(images, window_size_blur), images)
            images = multiplexing(condition_hsv, dali_random_hsv(images, hsv_hue, hsv_saturation), images)
            images = dali_random_gray(images, 0.1)

        images = fn.crop_mirror_normalize(
            images, dtype=types.FLOAT, mean=mean, std=std, mirror=condition_flip)
        pipe.set_outputs(images, labels)
    pipe.build()
    return DALIWarper(DALIClassificationIterator(pipelines=[pipe], reader_name=name, ))


@torch.no_grad()
class DALIWarper(object):
    def __init__(self, dali_iter):
        self.iter = dali_iter

    def __next__(self):
        data_dict = self.iter.__next__()[0]
        tensor_data = data_dict['data'].cuda()
        tensor_label: torch.Tensor = data_dict['label'].cuda().long()
        tensor_label.squeeze_()
        return tensor_data, tensor_label

    def __iter__(self):
        return self

    def reset(self):
        self.iter.reset()

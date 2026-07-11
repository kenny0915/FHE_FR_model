# coding: utf-8

import argparse
import json
import timeit
from pathlib import Path

import cv2
import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import DataLoader, Dataset

from backbones import get_model


TARGET_BN_EPS_LAYERS = {
    'layer3.7.herpn.bn2',
    'layer3.8.herpn.bn2',
    'layer3.9.herpn.bn2',
    'layer3.10.herpn.bn2',
    'layer3.11.herpn.bn2',
    'layer3.12.herpn.bn2',
    'layer3.13.herpn.bn2',
}


def parse_args():
    parser = argparse.ArgumentParser(description='do TinyFace identification test')
    parser.add_argument('--model-prefix', default='', help='path to load model.')
    parser.add_argument('--data-dir', default='tinyface', type=str, help='TinyFace root directory.')
    parser.add_argument('--result-dir', default='.', type=str, help='directory for results.')
    parser.add_argument('--batch-size', default=256, type=int, help='image extraction batch size.')
    parser.add_argument('--eval-batch-size', default=64, type=int, help='probe batch size for metric computation.')
    parser.add_argument('--num-workers', default=4, type=int, help='image loading workers.')
    parser.add_argument('--network', default='r50', type=str, help='backbone name.')
    parser.add_argument('--job', default='insightface', type=str, help='job name.')
    parser.add_argument('--embedding-size', default=None, type=int, help='optional num_features override for get_model.')
    parser.add_argument('--device', default=None, type=str, help='default: cuda if available, otherwise cpu.')
    parser.add_argument('--no-flip-test', dest='flip_test', action='store_false', help='disable horizontal flip feature sum.')
    parser.add_argument('--no-normalize-features', dest='normalize_features', action='store_false',
                        help='disable L2 normalization before Euclidean/Cosine evaluation.')
    parser.add_argument('--metric', default='euclidean', choices=['euclidean', 'cosine'],
                        help='ranking metric. With normalized features, Euclidean and Cosine rank identically.')
    parser.add_argument('--reuse-features', action='store_true',
                        help='load existing MAT feature files from result-dir/job/features if present.')
    parser.add_argument('--extract-only', action='store_true', help='extract and save features without metric evaluation.')
    parser.add_argument(
        '--layer3-herpn-bn2-eps',
        default=None,
        type=float,
        help='override eps only for layer3.7-13.herpn.bn2 BatchNorm layers',
    )
    parser.set_defaults(flip_test=True, normalize_features=True)
    return parser.parse_args()


def override_target_bn_eps(model, eps):
    if eps is None:
        return

    found = []
    for name, module in model.named_modules():
        if name in TARGET_BN_EPS_LAYERS:
            if not isinstance(module, torch.nn.BatchNorm2d):
                raise TypeError(f'{name} is {type(module)}, expected BatchNorm2d')
            module.eps = eps
            found.append(name)

    missing = sorted(TARGET_BN_EPS_LAYERS.difference(found))
    if missing:
        raise ValueError(f'Missing target BN layers: {missing}')
    print(f'Overrode eps={eps} for BN layers: {sorted(found)}')


def clean_state_dict(state):
    if isinstance(state, dict):
        for key in ('state_dict', 'model', 'backbone'):
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break

    if not isinstance(state, dict):
        raise TypeError('Checkpoint must be a state_dict or contain one.')

    if all(key.startswith('module.') for key in state.keys()):
        state = {key[len('module.'):]: value for key, value in state.items()}
    return state


def load_checkpoint(path):
    try:
        return torch.load(path, map_location='cpu', weights_only=True)
    except TypeError:
        return torch.load(path, map_location='cpu')


def build_model(args, device):
    model_kwargs = {'dropout': 0, 'fp16': False}
    if args.embedding_size is not None:
        model_kwargs['num_features'] = args.embedding_size

    model = get_model(args.network, **model_kwargs)
    state = load_checkpoint(args.model_prefix)
    model.load_state_dict(clean_state_dict(state))
    override_target_bn_eps(model, args.layer3_herpn_bn2_eps)
    model.to(device)
    model.eval()

    if device.type == 'cuda' and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    return model


def matlab_string(value):
    while isinstance(value, np.ndarray):
        if value.shape == ():
            value = value.item()
        elif value.size == 1:
            value = value.reshape(-1)[0]
        else:
            value = ''.join(str(x) for x in value.reshape(-1))
            break
    if isinstance(value, bytes):
        value = value.decode('utf-8')
    return str(value)


def load_name_id_pairs(path, name_key, id_key):
    mat = sio.loadmat(path)
    names = [matlab_string(item) for item in mat[name_key].reshape(-1)]
    ids = mat[id_key].reshape(-1).astype(np.int64)
    if len(names) != len(ids):
        raise ValueError(f'{path} has {len(names)} names but {len(ids)} ids.')
    return names, ids


class TinyFaceDataset(Dataset):
    def __init__(self, image_dir, image_names, flip_test=True):
        self.image_dir = Path(image_dir)
        self.image_names = list(image_names)
        self.flip_test = flip_test

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        image_name = self.image_names[index]
        image_path = self.image_dir / image_name
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(str(image_path))

        image = cv2.resize(image, (112, 112), interpolation=cv2.INTER_LINEAR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.transpose(image, (2, 0, 1))

        if self.flip_test:
            image_flip = image[:, :, ::-1].copy()
            image = np.stack([image, image_flip], axis=0)

        return torch.from_numpy(image.copy())


def replace_nonfinite(name, values):
    nonfinite_mask = ~np.isfinite(values)
    nonfinite_count = int(np.count_nonzero(nonfinite_mask))
    if nonfinite_count == 0:
        return values

    if values.ndim == 1:
        bad_rows = np.where(nonfinite_mask)[0]
    else:
        bad_rows = np.where(np.any(nonfinite_mask, axis=1))[0]
    print('Warning: {} contains {} non-finite values across {} rows. '
          'Replacing them with 0. First bad rows: {}'.format(
              name, nonfinite_count, len(bad_rows), bad_rows[:10].tolist()))
    return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)


def safe_l2_normalize(name, features, eps=1e-12):
    features = replace_nonfinite(name, features)
    norms = np.sqrt(np.sum(features ** 2, axis=1, keepdims=True))
    bad_norm_rows = np.where((norms[:, 0] <= eps) | ~np.isfinite(norms[:, 0]))[0]
    if len(bad_norm_rows) > 0:
        print('Warning: {} has {} zero/non-finite norm rows. '
              'Leaving those rows as zero. First rows: {}'.format(
                  name, len(bad_norm_rows), bad_norm_rows[:10].tolist()))
        norms[bad_norm_rows] = 1.0
    return features / norms


@torch.no_grad()
def extract_features(model, image_dir, image_names, args, device, split_name):
    dataset = TinyFaceDataset(image_dir, image_names, flip_test=args.flip_test)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == 'cuda',
    )

    features = []
    total = len(dataset)
    start = timeit.default_timer()
    for batch_index, images in enumerate(loader):
        if args.flip_test:
            batch_size = images.shape[0]
            images = images.reshape(batch_size * 2, 3, 112, 112)

        images = images.to(device=device, dtype=torch.float32, non_blocking=True)
        images.div_(255).sub_(0.5).div_(0.5)

        batch_features = model(images)
        if isinstance(batch_features, (tuple, list)):
            batch_features = batch_features[0]
        if args.flip_test:
            batch_features = batch_features.reshape(batch_size, 2, -1).sum(dim=1)

        features.append(batch_features.cpu().numpy().astype(np.float32))

        done = min((batch_index + 1) * args.batch_size, total)
        if batch_index == 0 or done == total or (batch_index + 1) % 50 == 0:
            elapsed = timeit.default_timer() - start
            print(f'{split_name}: extracted {done}/{total} images in {elapsed:.2f}s')

    features = np.concatenate(features, axis=0)
    features = replace_nonfinite(f'{split_name}_features', features)
    if args.normalize_features:
        features = safe_l2_normalize(f'{split_name}_features', features)
    return features.astype(np.float32, copy=False)


def load_feature_mat(path, key):
    mat = sio.loadmat(path)
    if key not in mat:
        raise KeyError(f'{path} does not contain {key}')
    return mat[key].astype(np.float32, copy=False)


def save_feature_mats(feature_dir, gallery_features, probe_features, distractor_features):
    feature_dir.mkdir(parents=True, exist_ok=True)
    sio.savemat(feature_dir / 'gallery.mat', {'gallery_feature_map': gallery_features}, do_compression=False)
    sio.savemat(feature_dir / 'probe.mat', {'probe_feature_map': probe_features}, do_compression=False)
    sio.savemat(feature_dir / 'distractor.mat', {'distractor_feature_map': distractor_features}, do_compression=False)


def compute_ap_and_first_rank(good_indices, sorted_indices):
    good_indices = np.asarray(good_indices, dtype=np.int64)
    num_good = len(good_indices)
    if num_good == 0:
        return 0.0, None

    matches = np.isin(sorted_indices, good_indices)
    hit_ranks = np.flatnonzero(matches)
    if len(hit_ranks) == 0:
        return 0.0, None

    hit_counts = np.arange(1, len(hit_ranks) + 1, dtype=np.float64)
    precision = hit_counts / (hit_ranks + 1)
    previous_precision = np.ones_like(precision)
    non_first_rank = hit_ranks > 0
    previous_precision[non_first_rank] = (
        (hit_counts[non_first_rank] - 1) / hit_ranks[non_first_rank]
    )
    ap = np.sum((1.0 / num_good) * ((previous_precision + precision) / 2))
    return float(ap), int(hit_ranks[0])


def rank_gallery(gallery_features, probe_features, metric, gallery_norm=None):
    similarity = np.matmul(gallery_features, probe_features.T)
    if metric == 'cosine':
        return np.argsort(-similarity, axis=0, kind='mergesort')

    if gallery_norm is None:
        gallery_norm = np.sum(gallery_features ** 2, axis=1, keepdims=True)
    probe_norm = np.sum(probe_features ** 2, axis=1, keepdims=True).T
    distance = gallery_norm + probe_norm - 2 * similarity
    return np.argsort(distance, axis=0, kind='mergesort')


def evaluate_identification(gallery_features, gallery_ids, probe_features, probe_ids, args):
    ap_values = np.zeros(len(probe_ids), dtype=np.float32)
    first_ranks = np.full(len(probe_ids), -1, dtype=np.int64)

    gallery_id_to_indices = {}
    for index, identity in enumerate(gallery_ids):
        gallery_id_to_indices.setdefault(int(identity), []).append(index)

    gallery_norm = None
    if args.metric == 'euclidean':
        gallery_norm = np.sum(gallery_features ** 2, axis=1, keepdims=True)

    start = timeit.default_timer()
    for batch_start in range(0, len(probe_ids), args.eval_batch_size):
        batch_end = min(batch_start + args.eval_batch_size, len(probe_ids))
        sorted_indices = rank_gallery(
            gallery_features,
            probe_features[batch_start:batch_end],
            args.metric,
            gallery_norm=gallery_norm,
        )

        for local_index, probe_index in enumerate(range(batch_start, batch_end)):
            good_indices = gallery_id_to_indices.get(int(probe_ids[probe_index]), [])
            ap, first_rank = compute_ap_and_first_rank(good_indices, sorted_indices[:, local_index])
            ap_values[probe_index] = ap
            if first_rank is not None:
                first_ranks[probe_index] = first_rank

        elapsed = timeit.default_timer() - start
        print(f'Evaluated probes {batch_start + 1}-{batch_end}/{len(probe_ids)} in {elapsed:.2f}s')

    valid = first_ranks >= 0
    metrics = {
        'mAP': float(np.mean(ap_values)),
        'rank1': float(np.mean(valid & (first_ranks < 1))),
        'rank5': float(np.mean(valid & (first_ranks < 5))),
        'rank10': float(np.mean(valid & (first_ranks < 10))),
        'rank20': float(np.mean(valid & (first_ranks < 20))),
        'num_gallery': int(len(gallery_ids)),
        'num_probe': int(len(probe_ids)),
    }
    return metrics, ap_values, first_ranks


def write_metrics(save_dir, metrics, ap_values, first_ranks):
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / 'tinyface_metrics.json', 'w') as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
    np.save(save_dir / 'tinyface_ap.npy', ap_values)
    np.save(save_dir / 'tinyface_first_ranks.npy', first_ranks)

    print(
        'mAP = {mAP:.6f}, r1 precision = {rank1:.6f}, '
        'r5 precision = {rank5:.6f}, r10 precision = {rank10:.6f}, '
        'r20 precision = {rank20:.6f}'.format(**metrics)
    )


def main():
    args = parse_args()
    if not args.model_prefix and not args.reuse_features:
        raise ValueError('--model-prefix is required unless --reuse-features is set.')

    data_dir = Path(args.data_dir)
    test_dir = data_dir / 'Testing_Set'
    gallery_dir = test_dir / 'Gallery_Match'
    probe_dir = test_dir / 'Probe'
    distractor_dir = test_dir / 'Gallery_Distractor'

    gallery_names, gallery_ids = load_name_id_pairs(
        test_dir / 'gallery_match_img_ID_pairs.mat',
        'gallery_set',
        'gallery_ids',
    )
    probe_names, probe_ids = load_name_id_pairs(
        test_dir / 'probe_img_ID_pairs.mat',
        'probe_set',
        'probe_ids',
    )
    distractor_names = sorted(path.name for path in distractor_dir.glob('*.jpg'))
    if not distractor_names:
        raise ValueError(f'No distractor images found under {distractor_dir}')

    save_dir = Path(args.result_dir) / args.job
    feature_dir = save_dir / 'features'

    gallery_mat = feature_dir / 'gallery.mat'
    probe_mat = feature_dir / 'probe.mat'
    distractor_mat = feature_dir / 'distractor.mat'
    has_feature_cache = gallery_mat.exists() and probe_mat.exists() and distractor_mat.exists()
    if args.reuse_features and has_feature_cache:
        print(f'Loading cached features from {feature_dir}')
        gallery_features = load_feature_mat(gallery_mat, 'gallery_feature_map')
        probe_features = load_feature_mat(probe_mat, 'probe_feature_map')
        distractor_features = load_feature_mat(distractor_mat, 'distractor_feature_map')
    else:
        if args.reuse_features and not args.model_prefix:
            raise FileNotFoundError(f'Feature cache is incomplete under {feature_dir}')
        device_name = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
        device = torch.device(device_name)
        model = build_model(args, device)

        gallery_features = extract_features(model, gallery_dir, gallery_names, args, device, 'gallery')
        probe_features = extract_features(model, probe_dir, probe_names, args, device, 'probe')
        distractor_features = extract_features(model, distractor_dir, distractor_names, args, device, 'distractor')
        save_feature_mats(feature_dir, gallery_features, probe_features, distractor_features)
        with open(feature_dir / 'distractor_names.txt', 'w') as handle:
            handle.write('\n'.join(distractor_names) + '\n')

    print('Feature shapes: gallery {}, probe {}, distractor {}'.format(
        gallery_features.shape,
        probe_features.shape,
        distractor_features.shape,
    ))

    if args.extract_only:
        return

    gallery_features = np.concatenate([gallery_features, distractor_features], axis=0)
    distractor_ids = -100 * np.ones(len(distractor_features), dtype=np.int64)
    gallery_ids = np.concatenate([gallery_ids, distractor_ids], axis=0)

    if args.normalize_features:
        gallery_features = safe_l2_normalize('combined_gallery_features', gallery_features)
        probe_features = safe_l2_normalize('probe_features_for_eval', probe_features)

    metrics, ap_values, first_ranks = evaluate_identification(
        gallery_features,
        gallery_ids,
        probe_features,
        probe_ids,
        args,
    )
    write_metrics(save_dir, metrics, ap_values, first_ranks)


if __name__ == '__main__':
    main()

# coding: utf-8

import os
import csv
import pickle

import matplotlib
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import timeit
import sklearn
import argparse
import cv2
import numpy as np
import torch
from skimage import transform as trans
from backbones import get_model, iresnet50
from sklearn.metrics import roc_curve, auc

try:
    from prettytable import PrettyTable
except ImportError:
    class PrettyTable:
        def __init__(self, field_names):
            self.field_names = [str(x) for x in field_names]
            self.rows = []

        def add_row(self, row):
            self.rows.append([str(x) for x in row])

        def __str__(self):
            rows = [self.field_names] + self.rows
            widths = [max(len(row[i]) for row in rows) for i in range(len(self.field_names))]
            def fmt(row):
                return ' | '.join(value.ljust(widths[i]) for i, value in enumerate(row))
            return "\n".join([fmt(self.field_names), "-+-".join("-" * w for w in widths)] + [fmt(row) for row in self.rows])
from pathlib import Path

import sys
import warnings

sys.path.insert(0, "../")
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='do ijb test')
# general
parser.add_argument('--model-prefix', default='', help='path to load model.')
parser.add_argument('--image-path', default='', type=str, help='')
parser.add_argument('--result-dir', default='.', type=str, help='')
parser.add_argument('--batch-size', default=128, type=int, help='')
parser.add_argument('--network', default='iresnet50', type=str, help='')
parser.add_argument('--job', default='insightface', type=str, help='job name')
parser.add_argument('--target', default='IJBC', type=str, help='target, set to IJBC or IJBB')
parser.add_argument('--activation-debug-dir', default='', type=str,
                    help='directory to write ResNet activation input/output statistics; disabled when empty')
parser.add_argument('--activation-debug-batches', default=10, type=int,
                    help='number of forward_db batches to record when activation debugging is enabled; <=0 records all batches')
parser.add_argument('--skip-activation-plot', action='store_true',
                    help='skip writing the PreciseReLU alpha=10 vs trained PReLU comparison plot')
parser.add_argument('--max-images', default=0, type=int,
                    help='debug limiter: only evaluate the first N images when >0')
args = parser.parse_args()

target = args.target
model_path = args.model_prefix
image_path = args.image_path
result_dir = args.result_dir
gpu_id = None
use_norm_score = True  # if Ture, TestMode(N1)
use_detector_score = True  # if Ture, TestMode(D1)
use_flip_test = True  # if Ture, TestMode(F1)
job = args.job
batch_size = args.batch_size
activation_plot_written = False


class _THORPolynomialGELUFunction(torch.autograd.Function):
    @staticmethod
    def _polyval(x, coeffs):
        y = coeffs[-1].to(dtype=x.dtype, device=x.device)
        for coeff in coeffs[:-1].flip(0):
            y = y * x + coeff.to(dtype=x.dtype, device=x.device)
        return y

    @staticmethod
    def forward(ctx, x, p1_coeffs, p2_coeffs, input_scale):
        compute_x = x.float()
        p1 = p1_coeffs.float()
        p2 = p2_coeffs.float()
        scaled_x = compute_x / float(input_scale)
        p1_x = _THORPolynomialGELUFunction._polyval(scaled_x, p1)
        tanh_half = _THORPolynomialGELUFunction._polyval(p1_x, p2)
        return (compute_x * (0.5 + tanh_half)).to(dtype=x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        raise RuntimeError("THORPolynomialGELU is intended for inference in this eval script.")


class THORPolynomialGELU(torch.nn.Module):
    _P1_COEFFS = (
        -1.06240033e-05, 1.64454894e-04, -5.83533517e-04, -3.80912692e-04,
        2.24431193e-03, 8.92295204e-03, -1.05277477e-02, -1.91827040e-02,
        -2.04634786e-01, 4.54014410e-01, -5.40759203e-01, 5.67745523e+00,
        -1.36433727e+01, 1.82574621e+01, -8.48849601e+01, 1.28686741e+02,
        3.66720281e+02, -1.01400159e+03, -1.26278856e+02, 2.21728878e+03,
        -9.95421415e+02, -2.31059465e+03, 1.73583957e+03, 1.27394360e+03,
        -1.27836230e+03, -3.66781716e+02, 4.79663919e+02, 4.94610178e+01,
        -9.06754761e+01, -2.36515790e+00, 8.74311855e+00, 1.62838703e-02,
    )
    _P2_COEFFS = (
        -1.70270667e+02, 6.81076279e+01, 1.79197364e+03, -6.81621043e+02,
        -8.49256169e+03, 3.05629446e+03, 2.39579397e+04, -8.10435126e+03,
        -4.48145152e+04, 1.41297616e+04, 5.86197512e+04, -1.70371505e+04,
        -5.51326382e+04, 1.45532495e+04, 3.77866438e+04, -8.87673890e+03,
        -1.89514802e+04, 3.84972853e+03, 6.94169727e+03, -1.16901058e+03,
        -1.84658407e+03, 2.41693754e+02, 3.54452276e+02, -3.24499570e+01,
        -4.91918227e+01, 2.58122977e+00, 5.78392852e+00, -9.45171527e-02,
    )

    def __init__(self, input_scale=64.0):
        super().__init__()
        self.input_scale = float(input_scale)
        self.register_buffer("p1_coeffs", torch.tensor(tuple(reversed(self._P1_COEFFS)), dtype=torch.float32))
        self.register_buffer("p2_coeffs", 0.5 * torch.tensor(tuple(reversed(self._P2_COEFFS)), dtype=torch.float32))

    def forward(self, x):
        return _THORPolynomialGELUFunction.apply(
            x, self.p1_coeffs, self.p2_coeffs, self.input_scale
        )


class PreciseReLUAlpha10(torch.nn.Module):
    # Appendix A of "Precise Approximation of Convolutional Neural Networks":
    # r_alpha(x) = 0.5 * (x + x * (p3 o p2 o p1)(x)), alpha = 10.
    _P1_COEFFS = (
        -1.68048812248597e-47, 1.08541842577442e1,
        5.19213405604261e-46, -6.22833925211098e1,
        -1.67358715007438e-45, 1.14369227820443e2,
        1.15437076692363e-45, -6.28023496973074e1,
    )
    _P2_COEFFS = (
        7.86253562483970e-39, 4.13976170985111,
        -7.18241741649940e-38, -5.84997640211679,
        5.17878634442782e-38, 2.94376255659280,
        -9.33059743960049e-39, -4.54530437460152e-1,
    )
    _P3_COEFFS = (
        3.75374153583292e-39, 3.29956739043733,
        -1.04537140020889e-37, -7.84227260291355,
        4.18647895984231e-37, 1.28907764115564e1,
        -6.09510159540855e-37, -1.24917112584486e1,
        4.05475441247124e-37, 6.94167991428074,
        -1.26770087815848e-37, -2.04298067399942,
        1.52452197400636e-38, 2.46407138926031e-1,
    )

    def __init__(self):
        super().__init__()
        self.register_buffer("p1_coeffs", torch.tensor(self._P1_COEFFS, dtype=torch.float32))
        self.register_buffer("p2_coeffs", torch.tensor(self._P2_COEFFS, dtype=torch.float32))
        self.register_buffer("p3_coeffs", torch.tensor(self._P3_COEFFS, dtype=torch.float32))

    @staticmethod
    def _polyval(x, coeffs):
        y = coeffs[-1].to(dtype=x.dtype, device=x.device)
        for coeff in coeffs[:-1].flip(0):
            y = y * x + coeff.to(dtype=x.dtype, device=x.device)
        return y

    def forward(self, x):
        compute_x = x.float()
        p1_x = self._polyval(compute_x, self.p1_coeffs.float())
        p2_x = self._polyval(p1_x, self.p2_coeffs.float())
        p3_x = self._polyval(p2_x, self.p3_coeffs.float())
        out = 0.5 * (compute_x + compute_x * p3_x)
        return out.to(dtype=x.dtype)



def collect_prelu_slopes(module):
    slopes = []
    for child in module.modules():
        if isinstance(child, torch.nn.PReLU):
            slopes.append(child.weight.detach().float().cpu().reshape(-1))
    if not slopes:
        return torch.tensor([0.25], dtype=torch.float32)
    return torch.cat(slopes)


def write_activation_comparison_plot(model, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    slopes = collect_prelu_slopes(model)
    slope_mean = float(slopes.mean())
    slope_min = float(slopes.min())
    slope_max = float(slopes.max())
    x = torch.linspace(-2.0, 2.0, 4001)
    precise = PreciseReLUAlpha10().eval()(x).detach().cpu().numpy()
    x_np = x.numpy()
    relu = np.maximum(x_np, 0.0)
    prelu_mean = np.where(x_np >= 0.0, x_np, slope_mean * x_np)
    prelu_min = np.where(x_np >= 0.0, x_np, slope_min * x_np)
    prelu_max = np.where(x_np >= 0.0, x_np, slope_max * x_np)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, xlim, title in (
            (axes[0], (-1.0, 1.0), "Approximation range [-1, 1]"),
            (axes[1], (-2.0, 2.0), "Outside range behavior [-2, 2]")):
        ax.plot(x_np, precise, label="PreciseReLU alpha=10", linewidth=2.0)
        ax.plot(x_np, relu, label="ReLU", linestyle="--", linewidth=1.4)
        ax.plot(x_np, prelu_mean, label="trained PReLU mean slope", linewidth=1.4)
        ax.fill_between(x_np, prelu_min, prelu_max, alpha=0.18, label="trained PReLU min/max slope")
        ax.axvline(-1.0, color="gray", linestyle=":", linewidth=1.0)
        ax.axvline(1.0, color="gray", linestyle=":", linewidth=1.0)
        ax.set_xlim(*xlim)
        ax.set_ylim(-0.75, 2.25)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
        ax.set_title(title)
        ax.set_xlabel("activation input")
        ax.set_ylabel("activation output")
    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle("IResNet50 activation replacement: PreciseReLU alpha=10 vs original trained PReLU")
    fig.tight_layout()
    out_path = os.path.join(save_dir, "precise_relu_alpha10_vs_prelu.png")
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print("Saved activation comparison plot to {}".format(out_path))
    print("PReLU slope stats: min={:.6g}, mean={:.6g}, max={:.6g}, count={}".format(
        slope_min, slope_mean, slope_max, slopes.numel()))


class ActivationDebugRecorder:
    _next_id = 0

    def __init__(self, model, debug_dir, max_batches):
        self.debug_dir = debug_dir
        self.max_batches = int(max_batches)
        self.batch_index = 0
        self.handles = []
        self.saved_nonfinite = set()
        self.recorder_id = ActivationDebugRecorder._next_id
        ActivationDebugRecorder._next_id += 1
        os.makedirs(debug_dir, exist_ok=True)
        self.csv_path = os.path.join(debug_dir, "activation_stats.csv")
        write_header = not os.path.exists(self.csv_path)
        self.csv_file = open(self.csv_path, "a", newline="")
        self.fieldnames = [
            "recorder_id", "batch", "module", "kind", "shape", "numel",
            "finite", "nan", "posinf", "neginf", "finite_min", "finite_max",
            "finite_mean", "finite_std", "finite_absmax",
        ]
        self.writer = csv.DictWriter(self.csv_file, fieldnames=self.fieldnames)
        if write_header:
            self.writer.writeheader()
        for name, module in model.named_modules():
            if isinstance(module, (PreciseReLUAlpha10, THORPolynomialGELU)):
                self.handles.append(module.register_forward_hook(self._make_hook(name)))
        print("Recording activation stats for {} modules into {}".format(
            len(self.handles), self.csv_path))

    def _enabled(self):
        return self.max_batches <= 0 or self.batch_index < self.max_batches

    def _make_hook(self, name):
        def hook(module, inputs, output):
            if not self._enabled() or not inputs:
                return
            self._write_stats(name, "input", inputs[0])
            self._write_stats(name, "output", output)
        return hook

    def _write_stats(self, name, kind, tensor):
        if not torch.is_tensor(tensor):
            return
        with torch.no_grad():
            t = tensor.detach()
            finite_mask = torch.isfinite(t)
            finite_count = int(finite_mask.sum().item())
            nan_count = int(torch.isnan(t).sum().item())
            posinf_count = int(torch.isposinf(t).sum().item())
            neginf_count = int(torch.isneginf(t).sum().item())
            numel = int(t.numel())
            if finite_count:
                finite = t[finite_mask].float()
                finite_min = float(finite.min().item())
                finite_max = float(finite.max().item())
                finite_mean = float(finite.mean().item())
                finite_std = float(finite.std(unbiased=False).item())
                finite_absmax = float(finite.abs().max().item())
            else:
                finite_min = finite_max = finite_mean = finite_std = finite_absmax = float("nan")
            self.writer.writerow({
                "recorder_id": self.recorder_id,
                "batch": self.batch_index,
                "module": name,
                "kind": kind,
                "shape": "x".join(str(v) for v in t.shape),
                "numel": numel,
                "finite": finite_count,
                "nan": nan_count,
                "posinf": posinf_count,
                "neginf": neginf_count,
                "finite_min": finite_min,
                "finite_max": finite_max,
                "finite_mean": finite_mean,
                "finite_std": finite_std,
                "finite_absmax": finite_absmax,
            })
            if finite_count != numel:
                print("Non-finite {} detected at batch {}, module {}: nan={}, +inf={}, -inf={}".format(
                    kind, self.batch_index, name, nan_count, posinf_count, neginf_count))
                key = (name, kind)
                if key not in self.saved_nonfinite:
                    self.saved_nonfinite.add(key)
                    sample = t[:min(8, t.shape[0])].detach().float().cpu().numpy() if t.ndim > 0 else t.detach().float().cpu().numpy()
                    safe_name = name.replace(".", "_").replace("/", "_")
                    out_path = os.path.join(
                        self.debug_dir,
                        "nonfinite_{}_{}_batch{}.npz".format(safe_name, kind, self.batch_index),
                    )
                    np.savez_compressed(out_path, values=sample)
                    print("Saved first non-finite {} sample to {}".format(kind, out_path))
            self.csv_file.flush()

    def write_model_output_stats(self, tensor):
        if self._enabled():
            self._write_stats("model_output", "output", tensor)

    def step(self):
        self.batch_index += 1

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.csv_file.close()

def replace_resnet_activations_with_precise_relu(module):
    replaced = 0
    for name, child in module.named_children():
        if isinstance(child, (torch.nn.PReLU)):
            setattr(module, name, PreciseReLUAlpha10())
            replaced += 1
        else:
            replaced += replace_resnet_activations_with_precise_relu(child)
    return replaced


def replace_poolformer_gelu_with_thor(module):
    replaced = 0
    for name, child in module.named_children():
        if isinstance(child, THORPolynomialGELU):
            continue
        if isinstance(child, torch.nn.GELU):
            setattr(module, name, THORPolynomialGELU())
            replaced += 1
        else:
            replaced += replace_poolformer_gelu_with_thor(child)
    return replaced


class Embedding(object):
    def __init__(self, prefix, data_shape, batch_size=1):
        image_size = (112, 112)
        self.image_size = image_size
        weight = torch.load(prefix)
        network = "r50" if args.network == "iresnet50" else args.network
        if network == "r50":
            resnet = iresnet50(False, dropout=0, fp16=False).cuda()
        else:
            resnet = get_model(network, dropout=0, fp16=False).cuda()
        resnet.load_state_dict(weight)
        global activation_plot_written
        save_path = os.path.join(result_dir, args.job)
        if network == "r50" and not args.skip_activation_plot and not activation_plot_written:
            write_activation_comparison_plot(resnet, save_path)
            activation_plot_written = True
        if network == "r50":
            # replaced = replace_resnet_activations_with_precise_relu(resnet)
            # print("Replaced {} ReLU/PReLU activations with precise ReLU alpha=10 for {} inference.".format(
            #     replaced, network))
            print("Skipping activation replacement for {} inference.".format(network))
        elif network.startswith("poolformer"):
            replaced = replace_poolformer_gelu_with_thor(resnet)
            print("Replaced {} GELU activations with THOR polynomial GELU for {} inference.".format(
                replaced, network))
        self.activation_recorder = None
        if args.activation_debug_dir:
            self.activation_recorder = ActivationDebugRecorder(
                resnet, args.activation_debug_dir, args.activation_debug_batches)
        model = torch.nn.DataParallel(resnet)
        self.model = model
        self.model.eval()
        src = np.array([
            [30.2946, 51.6963],
            [65.5318, 51.5014],
            [48.0252, 71.7366],
            [33.5493, 92.3655],
            [62.7299, 92.2041]], dtype=np.float32)
        src[:, 0] += 8.0
        self.src = src
        self.batch_size = batch_size
        self.data_shape = data_shape

    def get(self, rimg, landmark):

        assert landmark.shape[0] == 68 or landmark.shape[0] == 5
        assert landmark.shape[1] == 2
        if landmark.shape[0] == 68:
            landmark5 = np.zeros((5, 2), dtype=np.float32)
            landmark5[0] = (landmark[36] + landmark[39]) / 2
            landmark5[1] = (landmark[42] + landmark[45]) / 2
            landmark5[2] = landmark[30]
            landmark5[3] = landmark[48]
            landmark5[4] = landmark[54]
        else:
            landmark5 = landmark
        tform = trans.SimilarityTransform()
        tform.estimate(landmark5, self.src)
        M = tform.params[0:2, :]
        img = cv2.warpAffine(rimg,
                             M, (self.image_size[1], self.image_size[0]),
                             borderValue=0.0)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_flip = np.fliplr(img)
        img = np.transpose(img, (2, 0, 1))  # 3*112*112, RGB
        img_flip = np.transpose(img_flip, (2, 0, 1))
        input_blob = np.zeros((2, 3, self.image_size[1], self.image_size[0]), dtype=np.uint8)
        input_blob[0] = img
        input_blob[1] = img_flip
        return input_blob

    @torch.no_grad()
    def forward_db(self, batch_data):
        imgs = torch.Tensor(batch_data).cuda()
        imgs.div_(255).sub_(0.5).div_(0.5)
        feat = self.model(imgs)
        if self.activation_recorder is not None:
            self.activation_recorder.write_model_output_stats(feat)
            self.activation_recorder.step()
        feat = feat.reshape([self.batch_size, 2 * feat.shape[1]])
        return feat.cpu().numpy()


# 将一个list尽量均分成n份，限制len(list)==n，份数大于原list内元素个数则分配空list[]
def divideIntoNstrand(listTemp, n):
    twoList = [[] for i in range(n)]
    for i, e in enumerate(listTemp):
        twoList[i % n].append(e)
    return twoList


def read_template_media_list(path):
    # ijb_meta = np.loadtxt(path, dtype=str)
    ijb_meta = pd.read_csv(path, sep=' ', header=None).values
    templates = ijb_meta[:, 1].astype(int)
    medias = ijb_meta[:, 2].astype(int)
    return templates, medias


# In[ ]:


def read_template_pair_list(path):
    # pairs = np.loadtxt(path, dtype=str)
    pairs = pd.read_csv(path, sep=' ', header=None).values
    # print(pairs.shape)
    # print(pairs[:, 0].astype(int))
    t1 = pairs[:, 0].astype(int)
    t2 = pairs[:, 1].astype(int)
    label = pairs[:, 2].astype(int)
    return t1, t2, label


# In[ ]:


def read_image_feature(path):
    with open(path, 'rb') as fid:
        img_feats = pickle.load(fid)
    return img_feats


# In[ ]:


def get_image_feature(img_path, files_list, model_path, epoch, gpu_id):
    batch_size = args.batch_size
    data_shape = (3, 112, 112)

    files = files_list
    print('files:', len(files))
    rare_size = len(files) % batch_size
    faceness_scores = []
    batch = 0
    img_feats = np.empty((len(files), 1024), dtype=np.float32)

    batch_data = np.empty((2 * batch_size, 3, 112, 112))
    embedding = Embedding(model_path, data_shape, batch_size)
    for img_index, each_line in enumerate(files[:len(files) - rare_size]):
        name_lmk_score = each_line.strip().split(' ')
        img_name = os.path.join(img_path, name_lmk_score[0])
        img = cv2.imread(img_name)
        lmk = np.array([float(x) for x in name_lmk_score[1:-1]],
                       dtype=np.float32)
        lmk = lmk.reshape((5, 2))
        input_blob = embedding.get(img, lmk)

        batch_data[2 * (img_index - batch * batch_size)][:] = input_blob[0]
        batch_data[2 * (img_index - batch * batch_size) + 1][:] = input_blob[1]
        if (img_index + 1) % batch_size == 0:
            print('batch', batch)
            img_feats[batch * batch_size:batch * batch_size +
                                         batch_size][:] = embedding.forward_db(batch_data)
            batch += 1
        faceness_scores.append(name_lmk_score[-1])

    batch_data = np.empty((2 * rare_size, 3, 112, 112))
    embedding = Embedding(model_path, data_shape, rare_size)
    for img_index, each_line in enumerate(files[len(files) - rare_size:]):
        name_lmk_score = each_line.strip().split(' ')
        img_name = os.path.join(img_path, name_lmk_score[0])
        img = cv2.imread(img_name)
        lmk = np.array([float(x) for x in name_lmk_score[1:-1]],
                       dtype=np.float32)
        lmk = lmk.reshape((5, 2))
        input_blob = embedding.get(img, lmk)
        batch_data[2 * img_index][:] = input_blob[0]
        batch_data[2 * img_index + 1][:] = input_blob[1]
        if (img_index + 1) % rare_size == 0:
            print('batch', batch)
            img_feats[len(files) -
                      rare_size:][:] = embedding.forward_db(batch_data)
            batch += 1
        faceness_scores.append(name_lmk_score[-1])
    faceness_scores = np.array(faceness_scores).astype(np.float32)
    # img_feats = np.ones( (len(files), 1024), dtype=np.float32) * 0.01
    # faceness_scores = np.ones( (len(files), ), dtype=np.float32 )
    return img_feats, faceness_scores


# In[ ]:


def image2template_feature(img_feats=None, templates=None, medias=None):
    # ==========================================================
    # 1. face image feature l2 normalization. img_feats:[number_image x feats_dim]
    # 2. compute media feature.
    # 3. compute template feature.
    # ==========================================================
    unique_templates = np.unique(templates)
    template_feats = np.zeros((len(unique_templates), img_feats.shape[1]))

    for count_template, uqt in enumerate(unique_templates):

        (ind_t,) = np.where(templates == uqt)
        face_norm_feats = img_feats[ind_t]
        face_medias = medias[ind_t]
        unique_medias, unique_media_counts = np.unique(face_medias,
                                                       return_counts=True)
        media_norm_feats = []
        for u, ct in zip(unique_medias, unique_media_counts):
            (ind_m,) = np.where(face_medias == u)
            if ct == 1:
                media_norm_feats += [face_norm_feats[ind_m]]
            else:  # image features from the same video will be aggregated into one feature
                media_norm_feats += [
                    np.mean(face_norm_feats[ind_m], axis=0, keepdims=True)
                ]
        media_norm_feats = np.array(media_norm_feats)
        # media_norm_feats = media_norm_feats / np.sqrt(np.sum(media_norm_feats ** 2, -1, keepdims=True))
        template_feats[count_template] = np.sum(media_norm_feats, axis=0)
        if count_template % 2000 == 0:
            print('Finish Calculating {} template features.'.format(
                count_template))
    # template_norm_feats = template_feats / np.sqrt(np.sum(template_feats ** 2, -1, keepdims=True))
    template_norm_feats = sklearn.preprocessing.normalize(template_feats)
    # print(template_norm_feats.shape)
    return template_norm_feats, unique_templates


# In[ ]:


def verification(template_norm_feats=None,
                 unique_templates=None,
                 p1=None,
                 p2=None):
    # ==========================================================
    #         Compute set-to-set Similarity Score.
    # ==========================================================
    template2id = np.zeros((max(unique_templates) + 1, 1), dtype=int)
    for count_template, uqt in enumerate(unique_templates):
        template2id[uqt] = count_template

    score = np.zeros((len(p1),))  # save cosine distance between pairs

    total_pairs = np.array(range(len(p1)))
    batchsize = 100000  # small batchsize instead of all pairs in one batch due to the memory limiation
    sublists = [
        total_pairs[i:i + batchsize] for i in range(0, len(p1), batchsize)
    ]
    total_sublists = len(sublists)
    for c, s in enumerate(sublists):
        feat1 = template_norm_feats[template2id[p1[s]]]
        feat2 = template_norm_feats[template2id[p2[s]]]
        similarity_score = np.sum(feat1 * feat2, -1)
        score[s] = similarity_score.flatten()
        if c % 10 == 0:
            print('Finish {}/{} pairs.'.format(c, total_sublists))
    return score


# In[ ]:
def verification2(template_norm_feats=None,
                  unique_templates=None,
                  p1=None,
                  p2=None):
    template2id = np.zeros((max(unique_templates) + 1, 1), dtype=int)
    for count_template, uqt in enumerate(unique_templates):
        template2id[uqt] = count_template
    score = np.zeros((len(p1),))  # save cosine distance between pairs
    total_pairs = np.array(range(len(p1)))
    batchsize = 100000  # small batchsize instead of all pairs in one batch due to the memory limiation
    sublists = [
        total_pairs[i:i + batchsize] for i in range(0, len(p1), batchsize)
    ]
    total_sublists = len(sublists)
    for c, s in enumerate(sublists):
        feat1 = template_norm_feats[template2id[p1[s]]]
        feat2 = template_norm_feats[template2id[p2[s]]]
        similarity_score = np.sum(feat1 * feat2, -1)
        score[s] = similarity_score.flatten()
        if c % 10 == 0:
            print('Finish {}/{} pairs.'.format(c, total_sublists))
    return score


def read_score(path):
    with open(path, 'rb') as fid:
        img_feats = pickle.load(fid)
    return img_feats


# # Step1: Load Meta Data

# In[ ]:

assert target == 'IJBC' or target == 'IJBB'

# =============================================================
# load image and template relationships for template feature embedding
# tid --> template id,  mid --> media id
# format:
#           image_name tid mid
# =============================================================
start = timeit.default_timer()
templates, medias = read_template_media_list(
    os.path.join('%s/meta' % image_path,
                 '%s_face_tid_mid.txt' % target.lower()))
stop = timeit.default_timer()
print('Time: %.2f s. ' % (stop - start))

# In[ ]:

# =============================================================
# load template pairs for template-to-template verification
# tid : template id,  label : 1/0
# format:
#           tid_1 tid_2 label
# =============================================================
start = timeit.default_timer()
p1, p2, label = read_template_pair_list(
    os.path.join('%s/meta' % image_path,
                 '%s_template_pair_label.txt' % target.lower()))
stop = timeit.default_timer()
print('Time: %.2f s. ' % (stop - start))

# # Step 2: Get Image Features

# In[ ]:

# =============================================================
# load image features
# format:
#           img_feats: [image_num x feats_dim] (227630, 512)
# =============================================================
start = timeit.default_timer()
img_path = '%s/loose_crop' % image_path
img_list_path = '%s/meta/%s_name_5pts_score.txt' % (image_path, target.lower())
img_list = open(img_list_path)
files = img_list.readlines()
if args.max_images > 0:
    files = files[:args.max_images]
    print('Debug max-images enabled: evaluating first {} images.'.format(len(files)))
# files_list = divideIntoNstrand(files, rank_size)
files_list = files

# img_feats
# for i in range(rank_size):
img_feats, faceness_scores = get_image_feature(img_path, files_list,
                                               model_path, 0, gpu_id)
stop = timeit.default_timer()
print('Time: %.2f s. ' % (stop - start))
print('Feature Shape: ({} , {}) .'.format(img_feats.shape[0],
                                          img_feats.shape[1]))
if args.max_images > 0:
    print('Debug max-images run complete after feature extraction; skipping template aggregation.')
    sys.exit(0)

# # Step3: Get Template Features

# In[ ]:

# =============================================================
# compute template features from image features.
# =============================================================
start = timeit.default_timer()
# ==========================================================
# Norm feature before aggregation into template feature?
# Feature norm from embedding network and faceness score are able to decrease weights for noise samples (not face).
# ==========================================================
# 1. FaceScore （Feature Norm）
# 2. FaceScore （Detector）

if use_flip_test:
    # concat --- F1
    # img_input_feats = img_feats
    # add --- F2
    img_input_feats = img_feats[:, 0:img_feats.shape[1] //
                                     2] + img_feats[:, img_feats.shape[1] // 2:]
else:
    img_input_feats = img_feats[:, 0:img_feats.shape[1] // 2]

if use_norm_score:
    img_input_feats = img_input_feats
else:
    # normalise features to remove norm information
    img_input_feats = img_input_feats / np.sqrt(
        np.sum(img_input_feats ** 2, -1, keepdims=True))

if use_detector_score:
    print(img_input_feats.shape, faceness_scores.shape)
    img_input_feats = img_input_feats * faceness_scores[:, np.newaxis]
else:
    img_input_feats = img_input_feats

template_norm_feats, unique_templates = image2template_feature(
    img_input_feats, templates, medias)
stop = timeit.default_timer()
print('Time: %.2f s. ' % (stop - start))

# # Step 4: Get Template Similarity Scores

# In[ ]:

# =============================================================
# compute verification scores between template pairs.
# =============================================================
start = timeit.default_timer()
score = verification(template_norm_feats, unique_templates, p1, p2)
stop = timeit.default_timer()
print('Time: %.2f s. ' % (stop - start))

# In[ ]:
save_path = os.path.join(result_dir, args.job)
# save_path = result_dir + '/%s_result' % target

if not os.path.exists(save_path):
    os.makedirs(save_path)

score_save_file = os.path.join(save_path, "%s.npy" % target.lower())
np.save(score_save_file, score)

# # Step 5: Get ROC Curves and TPR@FPR Table

# In[ ]:

files = [score_save_file]
methods = []
scores = []
for file in files:
    methods.append(Path(file).stem)
    scores.append(np.load(file))

methods = np.array(methods)
scores = dict(zip(methods, scores))
colormap = plt.get_cmap('Set2')
colours = dict(zip(methods, [colormap(i) for i in range(methods.shape[0])]))
x_labels = [10 ** -6, 10 ** -5, 10 ** -4, 10 ** -3, 10 ** -2, 10 ** -1]
tpr_fpr_table = PrettyTable(['Methods'] + [str(x) for x in x_labels])
fig = plt.figure()
for method in methods:
    fpr, tpr, _ = roc_curve(label, scores[method])
    roc_auc = auc(fpr, tpr)
    fpr = np.flipud(fpr)
    tpr = np.flipud(tpr)  # select largest tpr at same fpr
    plt.plot(fpr,
             tpr,
             color=colours[method],
             lw=1,
             label=('[%s (AUC = %0.4f %%)]' %
                    (method.split('-')[-1], roc_auc * 100)))
    tpr_fpr_row = []
    tpr_fpr_row.append("%s-%s" % (method, target))
    for fpr_iter in np.arange(len(x_labels)):
        _, min_index = min(
            list(zip(abs(fpr - x_labels[fpr_iter]), range(len(fpr)))))
        tpr_fpr_row.append('%.2f' % (tpr[min_index] * 100))
    tpr_fpr_table.add_row(tpr_fpr_row)
plt.xlim([10 ** -6, 0.1])
plt.ylim([0.3, 1.0])
plt.grid(linestyle='--', linewidth=1)
plt.xticks(x_labels)
plt.yticks(np.linspace(0.3, 1.0, 8, endpoint=True))
plt.xscale('log')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC on IJB')
plt.legend(loc="lower right")
fig.savefig(os.path.join(save_path, '%s.pdf' % target.lower()))
print(tpr_fpr_table)




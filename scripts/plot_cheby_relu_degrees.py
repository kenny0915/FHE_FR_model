#!/usr/bin/env python3
"""Plot degree-4/8/16 ChebyReLU approximations against ReLU."""

import argparse
import os
from pathlib import Path
import sys

import matplotlib
import torch

matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from eval.non_linear_replacement import ChebyReLU


def parse_args():
    parser = argparse.ArgumentParser(
        description='Compare ChebyReLU polynomial degrees on a symmetric interval')
    parser.add_argument('--interval', default=3.0, type=float,
                        help='plot and approximation interval is [-interval, interval]')
    parser.add_argument('--samples', default=6001, type=int,
                        help='number of uniformly spaced plot samples')
    parser.add_argument(
        '--output',
        default='docs/cheby_relu_degrees_4_8_16_interval_minus3_3.png',
        help='output image path')
    return parser.parse_args()


def main():
    args = parse_args()
    if args.interval <= 0.0:
        raise ValueError('--interval must be positive')
    if args.samples < 2:
        raise ValueError('--samples must be at least 2')

    inputs = torch.linspace(
        -args.interval, args.interval, args.samples, dtype=torch.float64)
    relu = torch.relu(inputs)
    approximations = {
        degree: ChebyReLU(
            input_scale=args.interval, degree=degree).eval()(inputs)
        for degree in (4, 8, 16)
    }

    x = inputs.numpy()
    target = relu.numpy()
    colors = {4: '#d55e00', 8: '#0072b2', 16: '#009e73'}
    fig, (curve_axis, error_axis) = plt.subplots(
        2, 1, figsize=(10, 8), sharex=True,
        gridspec_kw={'height_ratios': (2.2, 1.0)})

    curve_axis.plot(x, target, color='black', linewidth=2.4, label='ReLU')
    for degree, approximation in approximations.items():
        curve_axis.plot(
            x, approximation.numpy(), color=colors[degree], linewidth=1.7,
            label='ChebyReLU degree {}'.format(degree))
    curve_axis.axhline(0.0, color='0.55', linewidth=0.8)
    curve_axis.axvline(0.0, color='0.55', linewidth=0.8)
    curve_axis.set_ylabel('Output')
    curve_axis.set_title(
        'ChebyReLU approximations to ReLU on '
        '[-{0:g}, {0:g}]'.format(args.interval))
    curve_axis.grid(True, linestyle='--', linewidth=0.6, alpha=0.45)
    curve_axis.legend(loc='upper left')

    for degree, approximation in approximations.items():
        error = approximation.numpy() - target
        max_error = float(abs(error).max())
        error_axis.plot(
            x, error, color=colors[degree], linewidth=1.5,
            label='degree {} (max |error|={:.5f})'.format(
                degree, max_error))
    error_axis.axhline(0.0, color='black', linewidth=0.9)
    error_axis.axvline(0.0, color='0.55', linewidth=0.8)
    error_axis.set_xlim(-args.interval, args.interval)
    error_axis.set_xlabel('Input x')
    error_axis.set_ylabel('Approximation error')
    error_axis.grid(True, linestyle='--', linewidth=0.6, alpha=0.45)
    error_axis.legend(loc='upper right', fontsize=9)

    fig.tight_layout()
    output_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(args.output, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('Saved ChebyReLU comparison plot to {}'.format(args.output))


if __name__ == '__main__':
    main()

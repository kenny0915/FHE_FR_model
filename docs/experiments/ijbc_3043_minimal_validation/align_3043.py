"""Reproduce IJB-C 3043.jpg alignment using its recorded five landmarks.

Requires numpy, opencv-python-headless and scikit-image.
The landmarks in report.json apply only to the original 3043.jpg.
"""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from skimage import transform


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image', help='original IJB-C loose_crop/3043.jpg')
    parser.add_argument('--output', default='3043_aligned.png')
    args = parser.parse_args()
    report = json.loads(Path(__file__).with_name('report.json').read_text())
    if hashlib.sha256(Path(args.image).read_bytes()).hexdigest() != report['source_image_sha256']:
        raise ValueError('Expected the exact original 3043.jpg recorded in report.json')
    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(args.image)
    if list(image.shape[:2][::-1]) != report['original_size_width_height']:
        raise ValueError('Image size does not match the recorded original 3043.jpg')
    matrix = transform.SimilarityTransform()
    if not matrix.estimate(np.array(report['source_landmarks'], dtype=np.float32),
                           np.array(report['target_landmarks'], dtype=np.float32)):
        raise ValueError('Cannot estimate similarity transform')
    # Same defaults as eval_ijbc.Embedding.get: INTER_LINEAR, BORDER_CONSTANT=0.
    aligned_bgr = cv2.warpAffine(image, matrix.params[:2, :], (112, 112), borderValue=0.0)
    if not cv2.imwrite(args.output, aligned_bgr):
        raise OSError('Could not write aligned image')
    print(f'Saved aligned 112x112 image: {args.output}')


if __name__ == '__main__':
    main()

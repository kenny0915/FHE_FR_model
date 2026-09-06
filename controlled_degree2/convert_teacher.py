"""Convert the confirmed iResNet-50 teacher to a controlled quadratic student."""

from __future__ import annotations

import argparse
import copy

import torch
from torch.nn import functional as F

from controlled_degree2.model import (
    load_calibration,
    load_teacher,
    quadratic_modules,
    replace_prelu_with_quadratic,
    save_checkpoint,
    set_quadratic_schedule,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--dataset-root",
        default=None,
        help="optional RecordIO directory for a lightweight step-0 check",
    )
    parser.add_argument("--check-images", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


@torch.no_grad()
def step_zero_check(teacher, student, loader, image_limit: int, device):
    totals = {
        "images": 0,
        "deployable_nonfinite": 0,
        "clipped_nonfinite": 0,
        "deployable_cosine_sum": 0.0,
        "deployable_cosine_count": 0,
        "clipped_cosine_sum": 0.0,
    }
    for images, _labels in loader:
        images = images.to(device, non_blocking=True)
        target = teacher(images).float()

        set_quadratic_schedule(student, alpha=1.0, clip_eval=False)
        deployable = student(images).float()
        finite = torch.isfinite(deployable).all(dim=1)
        totals["deployable_nonfinite"] += int((~finite).sum())
        if finite.any():
            totals["deployable_cosine_sum"] += float(
                F.cosine_similarity(deployable[finite], target[finite], dim=1).sum()
            )
            totals["deployable_cosine_count"] += int(finite.sum())

        set_quadratic_schedule(student, clip_eval=True)
        clipped = student(images).float()
        clipped_finite = torch.isfinite(clipped).all(dim=1)
        totals["clipped_nonfinite"] += int((~clipped_finite).sum())
        if clipped_finite.any():
            totals["clipped_cosine_sum"] += float(
                F.cosine_similarity(clipped[clipped_finite], target[clipped_finite], dim=1).sum()
            )
        totals["images"] += images.shape[0]
        if totals["images"] >= image_limit:
            break

    set_quadratic_schedule(student, clip_eval=False)
    clipped_count = totals["images"] - totals["clipped_nonfinite"]
    report = {
        "images": totals["images"],
        "deployable_nonfinite": totals["deployable_nonfinite"],
        "clipped_nonfinite": totals["clipped_nonfinite"],
        "deployable_cosine": totals["deployable_cosine_sum"]
        / max(totals["deployable_cosine_count"], 1),
        "clipped_cosine": totals["clipped_cosine_sum"] / max(clipped_count, 1),
    }
    return report


def main():
    args = parse_args()
    device = torch.device(args.device)
    teacher = load_teacher(args.teacher, device=device).eval()
    student = copy.deepcopy(teacher).eval()
    calibration = load_calibration(args.calibration)
    replaced = replace_prelu_with_quadratic(student, calibration)
    student.to(device)
    if len(replaced) != 25:
        raise RuntimeError(f"expected 25 PReLU activations, replaced {len(replaced)}")

    report = None
    if args.dataset_root:
        from dataset import MXFaceDataset

        dataset = MXFaceDataset(args.dataset_root, local_rank=0)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=args.num_workers > 0,
        )
        report = step_zero_check(
            teacher, student, loader, args.check_images, device
        )
        print(
            "step-0 deployable: cosine={deployable_cosine:.4f}, "
            "nonfinite={deployable_nonfinite}/{images}".format(**report)
        )
        print(
            "step-0 clipped:    cosine={clipped_cosine:.4f}, "
            "nonfinite={clipped_nonfinite}/{images}".format(**report)
        )
        if report["clipped_nonfinite"]:
            raise FloatingPointError("clipped step-0 student is non-finite; refusing export")

    # The trainer starts alpha at zero and progresses layer-by-layer.  Alpha is
    # runtime state, not checkpoint state; eval always defaults to pure q(x).
    set_quadratic_schedule(student, alpha=1.0, clip_eval=False)
    save_checkpoint(
        args.out,
        student.cpu(),
        calibration,
        teacher_weights=args.teacher,
        extra={"origin": "controlled_degree2_conversion", "step0": report},
    )
    channels = sum(module.channels for module in quadratic_modules(student))
    print(f"wrote {args.out}: 25 direct quadratics, {channels} per-channel fits")


if __name__ == "__main__":
    main()

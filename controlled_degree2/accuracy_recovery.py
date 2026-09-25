"""Prepare accuracy-first recovery commands; execute only with --run on GPU host."""
import argparse
import shlex
import subprocess
import sys
from pathlib import Path


def training_command(args):
    command = [
        sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={args.gpus}",
        "-m", "controlled_degree2.train",
        "--student-init", args.checkpoint, "--teacher", args.teacher,
        "--dataset-root", args.dataset_root,
        "--output-dir", str(Path(args.output_root) / args.arm),
        "--canary-root", args.canary_root or args.dataset_root, "--canary-sets", "lfw,cplfw",
        "--epochs", "1" if args.smoke else "3", "--batch-size", "128", "--global-batch", "2048",
        "--precision", "fp32", "--swap-epochs", "0", "--save-every-epoch",
        "--seed", "20260925",
    ]
    if args.arm != "control":
        command += [
            "--freeze-batchnorm-stats", "--lr-at-512", "0.0001",
            "--hint-start", "0.1", "--hint-end", "0.03",
            "--beta", "0.01", "--causal-tail-beta", "0",
            "--operator-bound-weight", "0", "--penalty-warmup-epochs", "0.1",
            "--aug-lowres", "0.05", "--aug-photo", "0.05", "--aug-crop", "0.05",
            "--aug-stress", "0", "--aug-pathological", "0",
            "--tail-replay-fraction", "0", "--tail-replay-capacity", "0",
        ]
    if args.arm == "coefficients":
        command += ["--train-coefficients", "--coefficient-lr-multiplier", "0.1"]
    if args.smoke:
        command += ["--limit-batches", str(2 * (2048 // (128 * args.gpus))), "--log-every", "1"]
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("control", "fixed", "coefficients"), required=True)
    parser.add_argument("--checkpoint", default="work_dirs/controlled_degree2_tail_ms1mv3_20260907/progressive/student_best.pt")
    parser.add_argument("--teacher", default="work_dirs/ms1mv3_r50/model.pt")
    parser.add_argument("--dataset-root", default="ms1m-retinaface-t1")
    parser.add_argument("--canary-root", default=None)
    parser.add_argument("--output-root", default="work_dirs/accuracy_recovery_20260925")
    parser.add_argument("--gpus", type=int, default=4)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="two real accumulated updates and full canaries")
    args = parser.parse_args()
    if args.gpus < 1 or 2048 % (128 * args.gpus):
        parser.error("GPU count must divide global batch 2048 with microbatch 128")
    command = training_command(args)
    print(shlex.join(command), flush=True)
    if args.run:
        # Reserve the arm directory atomically to prevent accidental overwrite.
        for path in (args.checkpoint, args.teacher,
                     str(Path(args.dataset_root) / "train.rec"),
                     str(Path(args.dataset_root) / "train.idx"),
                     str(Path(args.canary_root or args.dataset_root) / "lfw.bin"),
                     str(Path(args.canary_root or args.dataset_root) / "cplfw.bin")):
            if not Path(path).is_file():
                raise FileNotFoundError(path)
        (Path(args.output_root) / args.arm).mkdir(parents=True, exist_ok=False)
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()

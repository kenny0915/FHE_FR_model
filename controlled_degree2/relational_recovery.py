"""MS1MV3-only, fixed-epoch relationship-distillation comparison."""
import argparse
from pathlib import Path
import shlex
import subprocess

from controlled_degree2.accuracy_recovery import training_command

WEIGHTS = {'pointwise': 0., 'relation01': .1, 'relation1': 1.}


def command_for(args):
    # Reuse the validated fixed-coefficient accuracy recipe unchanged.
    from argparse import Namespace
    settings = Namespace(**vars(args))
    settings.arm = 'fixed'
    command = training_command(settings)
    command[command.index('--output-dir') + 1] = str(Path(args.output_root) / args.arm)
    command[command.index('--epochs') + 1] = '1' if args.smoke else '6'
    command[command.index('--seed') + 1] = '20260929'
    command += ['--w-relational', str(WEIGHTS[args.arm]), '--relational-temperature', '.05']
    return command


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arm', choices=WEIGHTS, required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--teacher', required=True)
    p.add_argument('--dataset-root', required=True)
    p.add_argument('--canary-root', required=True)
    p.add_argument('--output-root', required=True)
    p.add_argument('--gpus', type=int, default=4)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--run', action='store_true')
    args = p.parse_args()
    if args.gpus < 1 or 2048 % (128 * args.gpus):
        p.error('GPU count must divide global batch 2048 / microbatch 128')
    command = command_for(args)
    print(shlex.join(command), flush=True)
    if args.run:
        for file in [Path(args.checkpoint), Path(args.teacher),
                     *[Path(args.dataset_root) / f for f in ('train.rec', 'train.idx')],
                     *[Path(args.canary_root) / f for f in ('lfw.bin', 'cplfw.bin')]]:
            if not file.is_file():
                raise FileNotFoundError(file)
        (Path(args.output_root) / args.arm).mkdir(parents=True, exist_ok=False)
        subprocess.run(command, check=True)


if __name__ == '__main__':
    main()

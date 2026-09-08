"""DDP trainer for the controlled direct degree-2 experiment.

The loss, progressive swap, range hinge, input coverage and best-checkpoint
gate match the stabilising run10 recipe.  Gradient accumulation keeps the
run10 global batch of 2048 on the project's four V100 GPUs.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import random
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from controlled_degree2 import losses
from controlled_degree2.augment import prepare_range_batch
from controlled_degree2.model import (
    collect_causal_tail_penalty,
    collect_range_stats,
    load_controlled_checkpoint,
    load_teacher,
    operator_bound_penalty,
    operator_bound_targets,
    quadratic_modules,
    save_checkpoint,
    scale_intervals,
    set_lam_reg_ratio,
    set_quadratic_schedule,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--student-init", required=True)
    parser.add_argument("--teacher", default=None)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume", default=None)

    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128, help="per GPU microbatch")
    parser.add_argument("--global-batch", type=int, default=2048)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--channels-last", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--lr-at-512", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--lr-warmup-epochs", type=float, default=0.5)
    parser.add_argument("--grad-clip", type=float, default=5.0)

    parser.add_argument("--w-embedding", type=float, default=1.0)
    parser.add_argument("--hint-start", type=float, default=1.0)
    parser.add_argument("--hint-end", type=float, default=0.3)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=10.0)
    parser.add_argument("--penalty-warmup-epochs", type=float, default=1.0)
    parser.add_argument("--swap-epochs", type=float, default=2.0)
    parser.add_argument("--swap-ramp", type=float, default=2.0)
    parser.add_argument("--lam-reg-ratio", type=float, default=0.6)

    parser.add_argument("--aug-lowres", type=float, default=0.2)
    parser.add_argument("--aug-photo", type=float, default=0.2)
    parser.add_argument("--aug-crop", type=float, default=0.1)
    parser.add_argument("--aug-stress", type=float, default=0.4)
    parser.add_argument("--aug-pathological", type=float, default=0.05)
    parser.add_argument("--tail-replay-fraction", type=float, default=0.25)
    parser.add_argument("--tail-replay-capacity", type=int, default=1024)
    parser.add_argument("--tail-replay-warmup-steps", type=int, default=100)
    parser.add_argument("--causal-tail-beta", type=float, default=1.0)
    parser.add_argument("--activation-guard-ratio", type=float, default=1.0)
    parser.add_argument(
        "--activation-lam-scale",
        type=float,
        default=1.0,
        help="widen every quadratic activation interval before MS1MV3 training",
    )
    parser.add_argument(
        "--activation-lam-scale-layer3",
        type=float,
        default=1.0,
        help="additional interval widening for layer3 activations",
    )
    parser.add_argument("--operator-bound-weight", type=float, default=1e-4)
    parser.add_argument("--operator-bound-margin", type=float, default=0.10)
    parser.add_argument("--adversarial-tail-fraction", type=float, default=0.0)
    parser.add_argument("--adversarial-tail-steps", type=int, default=0)
    parser.add_argument("--adversarial-tail-epsilon", type=float, default=0.25)
    parser.add_argument("--adversarial-tail-step-size", type=float, default=0.125)
    parser.add_argument("--adversarial-tail-beta", type=float, default=1.0)
    parser.add_argument("--adversarial-tail-warmup-steps", type=int, default=0)
    parser.add_argument(
        "--freeze-through-layer3",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "freeze stem and layers 1-3, including BatchNorm running statistics; "
            "only layer4 and the embedding head are optimized"
        ),
    )
    parser.add_argument(
        "--freeze-through-layer4",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "freeze every polynomial-bearing module through layer4, including "
            "BatchNorm running statistics; only the embedding head is optimized"
        ),
    )

    parser.add_argument("--canary-root", default=None)
    parser.add_argument("--canary-sets", default="lfw,cplfw")
    parser.add_argument("--canary-batch-size", type=int, default=256)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--max-nonfinite-streak", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--limit-batches",
        type=int,
        default=0,
        help="nonzero lightweight smoke/debug limit per epoch",
    )
    return parser.parse_args()


def distributed_context():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    if not torch.cuda.is_available():
        if world_size > 1:
            raise RuntimeError("multi-process controlled training requires CUDA/NCCL")
        device = torch.device("cpu")
    else:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    return rank, world_size, local_rank, device


FROZEN_THROUGH_LAYER3 = ("conv1", "bn1", "prelu", "layer1", "layer2", "layer3")
FROZEN_THROUGH_LAYER4 = FROZEN_THROUGH_LAYER3 + ("layer4",)


def freeze_through_layer3(model):
    """Freeze the numerically verified prefix, including BN running state."""
    frozen = []
    for name in FROZEN_THROUGH_LAYER3:
        module = model.get_submodule(name)
        module.requires_grad_(False)
        module.eval()
        frozen.append(name)
    return tuple(frozen)


def freeze_through_layer4(model):
    """Freeze all polynomial-bearing modules while leaving the head trainable."""
    frozen = list(freeze_through_layer3(model))
    model.layer4.requires_grad_(False)
    model.layer4.eval()
    frozen.append("layer4")
    return tuple(frozen)


def keep_frozen_modules_eval(model, names):
    for name in names:
        model.get_submodule(name).eval()


@torch.no_grad()
def snapshot_batchnorm_state(model):
    """Capture mutable BatchNorm buffers before a potentially rejected batch.

    BatchNorm updates its running statistics during the forward pass.  Merely
    skipping ``optimizer.step()`` therefore does not reject a non-finite batch:
    NaN/Inf activations can permanently poison a checkpoint's running state.
    The tensors are small compared with activations, so cloning them once per
    microbatch is a cheap transaction boundary.
    """
    snapshot = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            snapshot[name] = (
                None if module.running_mean is None else module.running_mean.clone(),
                None if module.running_var is None else module.running_var.clone(),
                None
                if module.num_batches_tracked is None
                else module.num_batches_tracked.clone(),
            )
    return snapshot


@torch.no_grad()
def restore_batchnorm_state(model, snapshot):
    """Roll back BatchNorm buffers captured by :func:`snapshot_batchnorm_state`."""
    for name, (running_mean, running_var, num_batches_tracked) in snapshot.items():
        module = model.get_submodule(name) if name else model
        if running_mean is not None:
            module.running_mean.copy_(running_mean)
        if running_var is not None:
            module.running_var.copy_(running_var)
        if num_batches_tracked is not None:
            module.num_batches_tracked.copy_(num_batches_tracked)


def make_adversarial_tail_batch(
    model,
    images,
    eligible_mask,
    target_name,
    *,
    fraction,
    steps,
    epsilon,
    step_size,
):
    """Maximize one polynomial input peak on a small MS1MV3 subset.

    This is a training-only projected-gradient search.  It keeps pixels in
    the normalized image domain and within ``epsilon`` of an already
    augmented MS1MV3 image.  BatchNorm buffers are transactional because the
    inner search must not change the exported model state.
    """
    adversarial_mask = torch.zeros(
        images.shape[0], dtype=torch.bool, device=images.device
    )
    count = min(
        int(round(images.shape[0] * float(fraction))),
        int(eligible_mask.sum()),
    )
    if count <= 0 or steps <= 0 or not target_name:
        return images, adversarial_mask

    eligible = eligible_mask.nonzero(as_tuple=False).flatten()
    chosen = eligible.index_select(
        0, torch.randperm(eligible.numel(), device=images.device)[:count]
    )
    base = images.index_select(0, chosen).detach()
    adversarial = (
        base + torch.empty_like(base).uniform_(-float(epsilon), float(epsilon))
    ).clamp(-1.0, 1.0)
    batchnorm_state = snapshot_batchnorm_state(model)
    try:
        for _ in range(int(steps)):
            adversarial = adversarial.detach().requires_grad_(True)
            model(adversarial)
            peak = model.get_submodule(target_name).last_sample_peak
            if peak is None or not bool(torch.isfinite(peak).all()):
                break
            gradient = torch.autograd.grad(
                peak.mean(), adversarial, only_inputs=True
            )[0]
            adversarial = adversarial + float(step_size) * gradient.sign()
            adversarial = torch.maximum(
                torch.minimum(adversarial, base + float(epsilon)),
                base - float(epsilon),
            ).clamp(-1.0, 1.0)
    finally:
        restore_batchnorm_state(model, batchnorm_state)

    output = images.clone()
    output.index_copy_(0, chosen, adversarial.detach())
    adversarial_mask[chosen] = True
    return output, adversarial_mask


def targeted_tail_penalty(model, target_name, sample_mask):
    if not target_name or not bool(sample_mask.any()):
        return next(model.parameters()).new_zeros(())
    penalty = model.get_submodule(target_name).last_sample_penalty
    if penalty is None:
        return next(model.parameters()).new_zeros(())
    return penalty[sample_mask].mean()


def belongs_to_frozen_module(name, frozen_names):
    return any(name == prefix or name.startswith(prefix + ".") for prefix in frozen_names)


def seed_everything(seed, rank):
    value = seed + rank
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def hint_names(model):
    return ["prelu"] + [
        name for name, module in model.named_modules() if type(module).__name__ == "IBasicBlock"
    ]


def causal_tail_names(model, frozen_names=()):
    """Return deployment-order polynomial guards that can receive gradients.

    Training clips every quadratic input for numerical safety.  Consequently,
    starting the causal scan at layer3 hides an earlier layer1/layer2 escape
    that would remain unclipped in deployment and seed a later recurrence.
    """
    return [
        module.name
        for module in quadratic_modules(model)
        if not belongs_to_frozen_module(module.name, frozen_names)
    ]


class CausalTailReplay:
    """Small GPU replay buffer retaining MS1MV3 rows with the worst tail ratio."""

    def __init__(self, capacity, fraction, device):
        self.capacity = int(capacity)
        self.fraction = float(fraction)
        self.device = device
        self.images = None
        self.scores = None

    def update(self, images, scores):
        if self.capacity <= 0 or images.numel() == 0:
            return
        images = images.detach().to(self.device).contiguous()
        scores = scores.detach().to(self.device, dtype=torch.float32).flatten()
        if images.shape[0] != scores.shape[0]:
            raise ValueError("tail replay images and scores must have matching rows")
        if self.images is not None:
            images = torch.cat((self.images, images), dim=0)
            scores = torch.cat((self.scores, scores), dim=0)
        keep = min(self.capacity, scores.numel())
        values, indices = torch.topk(scores, keep, largest=True, sorted=False)
        self.images = images.index_select(0, indices).clone()
        self.scores = values.clone()

    def mix(self, images):
        count = min(int(round(images.shape[0] * self.fraction)), images.shape[0])
        if count <= 0 or self.images is None or self.images.shape[0] == 0:
            return images
        indices = torch.randint(self.images.shape[0], (count,), device=self.device)
        mixed = images.clone()
        mixed[:count] = self.images.index_select(0, indices)
        return mixed


def attach_hints(model, names):
    store, handles = {}, []
    for name in names:
        handles.append(
            model.get_submodule(name).register_forward_hook(
                lambda _module, _inputs, output, name=name: store.__setitem__(name, output)
            )
        )
    return store, handles


def alpha_schedule(model, step, swap_steps, ramp_steps):
    names = [module.name for module in quadratic_modules(model)]
    return {
        name: losses.swap_alpha_at(step, index, len(names), swap_steps, ramp_steps)
        for index, name in enumerate(names)
    }


def checkpoint_extra(optimizer, epoch, step, best_score, config, include_optimizer=False):
    payload = {
        "origin": "controlled_degree2_distillation",
        "epoch": epoch,
        "step": step,
        "best_canary": best_score,
        "train_config": config,
    }
    if include_optimizer:
        payload["optimizer"] = optimizer.state_dict()
    return payload


def load_canaries(root, names):
    if not root:
        return None
    from eval.verification import load_bin

    return {
        name: load_bin(os.path.join(root, f"{name}.bin"), (112, 112)) for name in names
    }


@torch.no_grad()
def evaluate_canaries(model, canaries, batch_size):
    from eval.verification import test

    model.eval()
    set_quadratic_schedule(model, alpha=1.0, clip_eval=False)
    scores = {}
    try:
        for name, dataset in canaries.items():
            result = test(
                dataset,
                model,
                batch_size=batch_size,
                fail_on_nonfinite=True,
            )
            scores[name] = float(result[2])
    except FloatingPointError as error:
        print(f"canary rejected: {error}", flush=True)
        return None
    finally:
        model.train()
    return scores


def main():
    args = parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.global_batch <= 0:
        raise ValueError("epochs and batch sizes must be positive")
    for name in (
        "aug_crop",
        "aug_lowres",
        "aug_photo",
        "aug_stress",
        "aug_pathological",
    ):
        value = getattr(args, name)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"--{name.replace('_', '-')} must be in [0, 1]")
    if args.aug_pathological >= 1.0:
        raise ValueError("--aug-pathological must leave at least one distilled sample")
    if not 0.0 <= args.tail_replay_fraction <= 1.0:
        raise ValueError("--tail-replay-fraction must be in [0, 1]")
    if args.tail_replay_capacity < 0 or args.tail_replay_warmup_steps < 0:
        raise ValueError("tail replay capacity and warmup must be non-negative")
    if args.causal_tail_beta < 0 or args.operator_bound_weight < 0:
        raise ValueError("tail and operator-bound weights must be non-negative")
    if not 0.0 <= args.adversarial_tail_fraction <= 1.0:
        raise ValueError("--adversarial-tail-fraction must be in [0, 1]")
    if args.adversarial_tail_steps < 0 or args.adversarial_tail_warmup_steps < 0:
        raise ValueError("adversarial tail steps and warmup must be non-negative")
    if args.adversarial_tail_epsilon < 0 or args.adversarial_tail_step_size < 0:
        raise ValueError("adversarial tail epsilon and step size must be non-negative")
    if args.adversarial_tail_beta < 0:
        raise ValueError("adversarial tail weight must be non-negative")
    if args.activation_guard_ratio <= 0 or args.operator_bound_margin < 0:
        raise ValueError("activation guard ratio must be positive and margin non-negative")
    if args.activation_lam_scale <= 0:
        raise ValueError("activation lam scale must be positive")
    if args.activation_lam_scale_layer3 <= 0:
        raise ValueError("layer3 activation lam scale must be positive")
    if args.freeze_through_layer3 and args.freeze_through_layer4:
        raise ValueError("select only one frozen-prefix boundary")
    rank, world_size, local_rank, device = distributed_context()
    seed_everything(args.seed, rank)
    is_primary = rank == 0

    if args.global_batch % (args.batch_size * world_size):
        raise ValueError(
            "--global-batch must be divisible by per-GPU batch-size * WORLD_SIZE"
        )
    accumulation = args.global_batch // (args.batch_size * world_size)
    if accumulation < 1:
        raise ValueError("global batch is smaller than the distributed microbatch")

    student, init_payload = load_controlled_checkpoint(args.student_init, device=device)
    teacher_path = args.teacher or init_payload.get("teacher_weights")
    if not teacher_path:
        raise ValueError("--teacher is required when student-init does not record it")
    teacher = load_teacher(teacher_path, device=device).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    calibration = init_payload["poly_calib"]
    if args.activation_lam_scale != 1.0:
        scale_intervals(
            student,
            calibration,
            {"all": args.activation_lam_scale},
        )
    if args.activation_lam_scale_layer3 != 1.0:
        scale_intervals(
            student,
            calibration,
            {"layer3": args.activation_lam_scale_layer3},
        )
    set_lam_reg_ratio(student, args.lam_reg_ratio)
    if args.freeze_through_layer4:
        frozen_names = freeze_through_layer4(student)
    elif args.freeze_through_layer3:
        frozen_names = freeze_through_layer3(student)
    else:
        frozen_names = ()
    operator_targets = operator_bound_targets(student, args.operator_bound_margin)
    for entry in calibration.values():
        entry["lam_reg"] = (
            np.asarray(entry["lam_fit"], dtype=np.float64) * args.lam_reg_ratio
        ).tolist()

    if args.channels_last:
        student.to(memory_format=torch.channels_last)
        teacher.to(memory_format=torch.channels_last)

    from dataset import MXFaceDataset

    dataset = MXFaceDataset(args.dataset_root, local_rank=local_rank)
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=args.seed,
        drop_last=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=4 if args.num_workers > 0 else None,
    )
    microbatches_per_epoch = len(loader)
    if args.limit_batches:
        microbatches_per_epoch = min(microbatches_per_epoch, args.limit_batches)
    steps_per_epoch = microbatches_per_epoch // accumulation
    if steps_per_epoch < 1:
        raise ValueError("not enough batches for one accumulated optimizer step")
    total_steps = steps_per_epoch * args.epochs
    swap_steps = round(steps_per_epoch * args.swap_epochs)
    ramp_steps = max(
        1,
        round(args.swap_ramp * swap_steps / max(sum(1 for _ in quadratic_modules(student)), 1)),
    )
    penalty_warmup = round(steps_per_epoch * args.penalty_warmup_epochs)
    lr_warmup = round(steps_per_epoch * args.lr_warmup_epochs)
    learning_rate = args.lr_at_512 * args.global_batch / 512.0

    names = [
        name for name in hint_names(student)
        if not belongs_to_frozen_module(name, frozen_names)
    ]
    causal_names = causal_tail_names(student, frozen_names)
    student_hints, student_handles = attach_hints(student, names)
    teacher_hints, teacher_handles = attach_hints(teacher, names)
    if world_size > 1:
        distributed_student = DistributedDataParallel(
            student, device_ids=[local_rank], output_device=local_rank
        )
    else:
        distributed_student = student
    trainable = [parameter for parameter in student.parameters() if parameter.requires_grad]
    optimizer = torch.optim.SGD(
        trainable,
        lr=learning_rate,
        momentum=0.9,
        weight_decay=args.weight_decay,
        nesterov=True,
    )

    start_epoch, step, best_score = 0, 0, -1.0
    if args.resume:
        resume = torch.load(args.resume, map_location="cpu", weights_only=False)
        student.load_state_dict(resume["state_dict_backbone"], strict=True)
        optimizer.load_state_dict(resume["optimizer"])
        start_epoch = int(resume["epoch"]) + 1
        step = int(resume["step"])
        best_score = float(resume.get("best_canary", -1.0))

    canary_names = [name for name in args.canary_sets.split(",") if name]
    canaries = load_canaries(args.canary_root, canary_names) if is_primary else None
    os.makedirs(args.output_dir, exist_ok=True)
    config = vars(args) | {
        "world_size": world_size,
        "gradient_accumulation": accumulation,
        "effective_global_batch": args.batch_size * world_size * accumulation,
        "learning_rate": learning_rate,
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps,
        "degree": 2,
        "reference_ranges": "exact run10 checkpoint buffers",
        "causal_tail_layers": causal_names,
        "operator_bound_proxy": "max convolution row Frobenius norm",
        "activation_lam_scale": args.activation_lam_scale,
        "activation_lam_scale_layer3": args.activation_lam_scale_layer3,
        "frozen_modules": frozen_names,
    }
    if is_primary:
        with open(
            os.path.join(args.output_dir, "config.json"), "w", encoding="utf-8"
        ) as handle:
            json.dump(config, handle, indent=2)
        print(
            f"{world_size} GPU(s), microbatch {args.batch_size}, accumulate {accumulation}, "
            f"global batch {args.global_batch}, lr {learning_rate:.2e}"
        )
        print(
            f"{steps_per_epoch} optimizer steps/epoch; progressive swap completes near "
            f"step {swap_steps + ramp_steps}; {len(names)} hint points"
        )

    autocast_enabled = device.type == "cuda" and args.precision == "bf16"
    nonfinite_streak = 0
    replay = CausalTailReplay(args.tail_replay_capacity, args.tail_replay_fraction, device)
    optimizer.zero_grad(set_to_none=True)
    started = time.time()

    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        distributed_student.train()
        keep_frozen_modules_eval(student, frozen_names)
        consumed = 0
        for batch_index, (images, _labels) in enumerate(loader):
            if batch_index >= microbatches_per_epoch:
                break
            # Ignore a trailing partial accumulation so all optimizer steps see
            # exactly the declared global batch.
            if batch_index >= steps_per_epoch * accumulation:
                break
            images = images.to(device, non_blocking=True)
            if args.channels_last:
                images = images.contiguous(memory_format=torch.channels_last)
            images, distill_mask = prepare_range_batch(
                images,
                pathological_fraction=args.aug_pathological,
                crop_probability=args.aug_crop,
                lowres_probability=args.aug_lowres,
                photo_probability=args.aug_photo,
                stress_probability=args.aug_stress,
            )
            if step >= args.tail_replay_warmup_steps:
                images = replay.mix(images)
            mask = None if bool(distill_mask.all()) else distill_mask

            gamma = losses.gamma_at(step, penalty_warmup, args.gamma)
            beta = losses.beta_at(step, penalty_warmup, args.beta)
            hint_weight = losses.hint_weight_at(
                step, total_steps, args.hint_start, args.hint_end
            )
            alphas = alpha_schedule(student, step, swap_steps, ramp_steps)
            set_quadratic_schedule(
                student,
                alpha=alphas,
                gamma=gamma,
                clip=True,
                penalty="hinge",
                causal_guard_ratio=args.activation_guard_ratio,
            )
            adversarial_target = None
            adversarial_mask = torch.zeros_like(distill_mask)
            if (
                args.adversarial_tail_fraction > 0
                and args.adversarial_tail_steps > 0
                and step >= args.adversarial_tail_warmup_steps
                and causal_names
            ):
                target_index = (
                    (epoch * microbatches_per_epoch + batch_index) * world_size + rank
                ) % len(causal_names)
                adversarial_target = causal_names[target_index]
                images, adversarial_mask = make_adversarial_tail_batch(
                    student,
                    images,
                    distill_mask,
                    adversarial_target,
                    fraction=args.adversarial_tail_fraction,
                    steps=args.adversarial_tail_steps,
                    epsilon=args.adversarial_tail_epsilon,
                    step_size=args.adversarial_tail_step_size,
                )

            final_microbatch = (batch_index + 1) % accumulation == 0
            sync_context = (
                contextlib.nullcontext()
                if final_microbatch or world_size == 1
                else distributed_student.no_sync()
            )
            with sync_context:
                batchnorm_state = snapshot_batchnorm_state(student)
                with torch.no_grad(), torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=autocast_enabled,
                ):
                    teacher_embedding = teacher(images)
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=autocast_enabled,
                ):
                    student_embedding = distributed_student(images)
                    embedding = losses.embedding_loss(
                        student_embedding, teacher_embedding, mask
                    )
                    hint = (
                        losses.hint_loss(student_hints, teacher_hints, names, mask)
                        if names else student_embedding.new_zeros(())
                    )
                    if names:
                        range_penalty, oor, maxima = collect_range_stats(student)
                    else:
                        range_penalty = student_embedding.new_zeros(())
                        oor, maxima = {}, {}
                    causal_penalty = collect_causal_tail_penalty(
                        student,
                        causal_names,
                        guard_ratio=args.activation_guard_ratio,
                        # Pathological rows are excluded from teacher
                        # distillation, not from the MS1MV3-independent tail
                        # safety objective they were generated to exercise.
                        sample_mask=None,
                    )
                    bound_penalty = operator_bound_penalty(student, operator_targets)
                    adversarial_penalty = targeted_tail_penalty(
                        student, adversarial_target, adversarial_mask
                    )
                    loss = (
                        args.w_embedding * embedding
                        + hint_weight * hint
                        + beta * range_penalty
                        + args.causal_tail_beta * causal_penalty
                        + args.operator_bound_weight * bound_penalty
                        + args.adversarial_tail_beta * adversarial_penalty
                    )

                finite = torch.tensor(
                    int(torch.isfinite(loss)), dtype=torch.int32, device=device
                )
                if world_size > 1:
                    dist.all_reduce(finite, op=dist.ReduceOp.MIN)
                if not bool(finite):
                    # Forward-time BatchNorm updates are state mutations too.
                    # Roll them back on every rank when any rank rejects the
                    # batch, otherwise a skipped tail sample can silently make
                    # the next exported checkpoint non-deployable.
                    restore_batchnorm_state(student, batchnorm_state)
                    optimizer.zero_grad(set_to_none=True)
                    nonfinite_streak += 1
                    if is_primary:
                        worst = max(maxima, key=maxima.get) if maxima else "frozen-poly"
                        print(
                            f"non-finite loss before step {step}; worst {worst}="
                            f"{maxima.get(worst, 0.0):.3g}; skipped ({nonfinite_streak}/"
                            f"{args.max_nonfinite_streak})",
                            flush=True,
                        )
                    if nonfinite_streak >= args.max_nonfinite_streak:
                        raise FloatingPointError("too many consecutive non-finite batches")
                    continue
                (loss / accumulation).backward()
                nonfinite_streak = 0

            consumed += 1
            if not final_microbatch:
                continue

            torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
            lr_scale = losses.lr_factor(step, lr_warmup, total_steps)
            for group in optimizer.param_groups:
                group["lr"] = learning_rate * lr_scale
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1

            if replay.capacity > 0 and bool(distill_mask.any()):
                tail_ratios = [
                    module.last_sample_ratio
                    for module in quadratic_modules(student)
                    if module.name in causal_names and module.last_sample_ratio is not None
                ]
                if tail_ratios:
                    tail_scores = torch.stack(tail_ratios, dim=1).amax(dim=1)
                    replay.update(images[distill_mask], tail_scores[distill_mask])

            if is_primary and step % args.log_every == 0:
                worst = max(maxima, key=maxima.get) if maxima else "frozen-poly"
                elapsed = max(time.time() - started, 1e-6)
                throughput = step * args.global_batch / elapsed
                print(
                    f"step {step:>6} loss={float(loss):.4f} "
                    f"emb={float(embedding):.4f} hint={float(hint):.4f} "
                    f"pen={float(range_penalty):.3g} oor={np.mean(list(oor.values())):.2e} "
                            f"tail={float(causal_penalty):.3g} bound={float(bound_penalty):.3g} "
                            f"adv={float(adversarial_penalty):.3g}@{adversarial_target or '-'} "
                    f"max/lam={maxima.get(worst, 0.0):.2f}@{worst} "
                    f"poly={sum(a >= 1 for a in alphas.values())}/25 {throughput:.0f} img/s",
                    flush=True,
                )

        if world_size > 1:
            dist.barrier()
        if is_primary:
            extra = checkpoint_extra(
                optimizer, epoch, step, best_score, config, include_optimizer=True
            )
            save_checkpoint(
                os.path.join(args.output_dir, "last.pt"),
                student,
                calibration,
                teacher_weights=teacher_path,
                extra=extra,
            )
            canary_scores = (
                evaluate_canaries(student, canaries, args.canary_batch_size)
                if canaries
                else None
            )
            fully_polynomial = min(
                alpha_schedule(student, step, swap_steps, ramp_steps).values()
            ) >= 1.0
            if canary_scores is not None:
                score = float(np.mean(list(canary_scores.values())))
                print(f"epoch {epoch + 1} canary={canary_scores}, mean={score:.6f}")
                if fully_polynomial and score > best_score:
                    best_score = score
                    extra = checkpoint_extra(optimizer, epoch, step, best_score, config)
                    extra["canary"] = canary_scores
                    extra["canary_nonfinite"] = 0
                    save_checkpoint(
                        os.path.join(args.output_dir, "student_best.pt"),
                        student,
                        calibration,
                        teacher_weights=teacher_path,
                        extra=extra,
                    )
                    print("exported deployable student_best.pt")
            elif fully_polynomial:
                print(
                    "no --canary-root: last.pt saved, but student_best.pt is intentionally "
                    "not exported because the finite deployable gate was not run"
                )
        if world_size > 1:
            dist.barrier()

    if is_primary:
        save_checkpoint(
            os.path.join(args.output_dir, "student_final.pt"),
            student,
            calibration,
            teacher_weights=teacher_path,
            extra=checkpoint_extra(optimizer, args.epochs - 1, step, best_score, config),
        )
        print("exported student_final.pt")
    for handle in student_handles + teacher_handles:
        handle.remove()
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

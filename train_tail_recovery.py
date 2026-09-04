"""Numerically constrained full-backbone recovery for a converted HerPN R50.

This trainer intentionally has a narrow scope.  It separates ordinary
MS1Mv3/teacher updates from exact rare-tail containment updates, uses clean
batches alone to update BatchNorm buffers, and manually averages the combined
conflict-aware gradients across ranks.
"""

import argparse
import logging
import os
import tempfile

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributed
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

from backbones import get_model
from dataset import DatasetWithIndex, get_dataloader
from lr_scheduler import CosineLRWarmup
from utils.utils_config import get_config
from utils.utils_distributed_sampler import setup_seed
from utils.utils_logging import init_logging
from utils.utils_multi_objective import (
    combine_conflict_aware_gradients,
    project_to_relative_trust_region,
    synchronize_and_assign_gradients,
    temporary_batchnorm_eval,
)
from utils.utils_tail_recovery import load_fixed_tail_replay_orientations


def atomic_torch_save(value, path):
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    descriptor, temporary_path = tempfile.mkstemp(
        prefix=".tail-recovery-", suffix=".pt", dir=directory)
    os.close(descriptor)
    try:
        torch.save(value, temporary_path)
        os.replace(temporary_path, path)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)


def load_backbone_checkpoint(model, path):
    checkpoint = torch.load(path, map_location="cpu")
    state = checkpoint
    if isinstance(checkpoint, dict):
        for key in ("state_dict_backbone", "state_dict", "model"):
            if isinstance(checkpoint.get(key), dict):
                state = checkpoint[key]
                break
    state = {
        (name[7:] if name.startswith("module.") else name): tensor
        for name, tensor in state.items()
    }
    model.load_state_dict(state, strict=True)


def select_recovery_parameters(model):
    """Train Conv, ordinary BN affine, and HerPN coefficients, but not FC."""
    selected_names = []
    selected_parameters = []
    for name, parameter in model.named_parameters():
        trainable = not (
            name.startswith("fc.")
            or name.startswith("features.")
            or name.endswith(".prelu.weight")
        )
        parameter.requires_grad_(trainable)
        if trainable:
            selected_names.append(name)
            selected_parameters.append(parameter)
    if not selected_parameters:
        raise RuntimeError("Tail recovery selected no trainable parameters")
    return tuple(selected_names), tuple(selected_parameters)


def snapshot_batchnorm_buffers(model):
    snapshots = []
    for module in model.modules():
        if not isinstance(module, nn.modules.batchnorm._BatchNorm):
            continue
        snapshots.append((
            module,
            None if module.running_mean is None
            else module.running_mean.detach().clone(),
            None if module.running_var is None
            else module.running_var.detach().clone(),
            None if module.num_batches_tracked is None
            else module.num_batches_tracked.detach().clone(),
        ))
    return snapshots


@torch.no_grad()
def restore_batchnorm_buffers(snapshots):
    for module, running_mean, running_var, batches in snapshots:
        if running_mean is not None:
            module.running_mean.copy_(running_mean)
        if running_var is not None:
            module.running_var.copy_(running_var)
        if batches is not None:
            module.num_batches_tracked.copy_(batches)


def all_ranks_true(value, device):
    result = torch.tensor(bool(value), device=device, dtype=torch.long)
    if distributed.is_initialized():
        distributed.all_reduce(result, op=distributed.ReduceOp.MIN)
    return bool(result.item())


def gradients_are_finite(*gradient_groups):
    finite = [
        torch.isfinite(gradient).all()
        for gradients in gradient_groups
        for gradient in gradients
        if gradient is not None
    ]
    return not finite or bool(torch.stack(finite).all().item())


def make_tail_loader(cfg, train_dataset, rank, world_size, shuffle):
    orientations = load_fixed_tail_replay_orientations(
        cfg.fixed_tail_replay_file,
        cfg.fixed_tail_replay_orientations_key,
    )
    oriented_dataset = DatasetWithIndex(train_dataset, both_orientations=True)
    indices = tuple(
        2 * source_index + orientation
        for source_index, orientation in orientations
    )
    subset = Subset(oriented_dataset, indices)
    sampler = DistributedSampler(
        subset,
        num_replicas=world_size,
        rank=rank,
        shuffle=shuffle,
        seed=int(cfg.seed),
        drop_last=False,
    )
    loader = DataLoader(
        subset,
        batch_size=int(cfg.fixed_tail_replay_batch_size),
        sampler=sampler,
        num_workers=int(cfg.fixed_tail_replay_workers),
        pin_memory=True,
        drop_last=False,
        persistent_workers=int(cfg.fixed_tail_replay_workers) > 0,
    )
    return loader, sampler, len(orientations)


def next_cycled(iterator, loader, sampler, cycle):
    try:
        return next(iterator), iterator, cycle
    except StopIteration:
        cycle += 1
        sampler.set_epoch(cycle)
        iterator = iter(loader)
        return next(iterator), iterator, cycle


@torch.no_grad()
def exact_tail_gate(model, loader, device):
    model.eval()
    local_total = 0
    local_nonfinite = 0
    for batch in loader:
        images = batch[0].to(device=device, non_blocking=True)
        embeddings = model(images)
        finite = torch.isfinite(embeddings).all(dim=1)
        local_total += int(finite.numel())
        local_nonfinite += int((~finite).sum().item())
    counts = torch.tensor(
        [local_nonfinite, local_total], device=device, dtype=torch.long)
    distributed.all_reduce(counts, op=distributed.ReduceOp.SUM)
    return int(counts[0].item()), int(counts[1].item())


def main(config_path):
    distributed.init_process_group("nccl")
    rank = distributed.get_rank()
    world_size = distributed.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    cfg = get_config(config_path)
    setup_seed(int(cfg.seed), cuda_deterministic=False)
    os.makedirs(cfg.output, exist_ok=True)
    init_logging(rank, cfg.output)

    if cfg.network != "r50_no_relu":
        raise ValueError("This recovery trainer currently requires r50_no_relu")
    if cfg.fp16:
        raise ValueError("Conflict-aware recovery requires FP32")
    if cfg.optimizer != "sgd" or float(cfg.momentum) != 0.0:
        raise ValueError("Conflict-aware recovery requires zero-momentum SGD")

    train_loader = get_dataloader(
        cfg.rec,
        local_rank,
        cfg.batch_size,
        cfg.dali,
        cfg.dali_aug,
        cfg.seed,
        cfg.num_workers,
        range_augmentation=None,
    )
    tail_loader, tail_sampler, tail_count = make_tail_loader(
        cfg, train_loader.dataset, rank, world_size, shuffle=True)
    tail_eval_loader, tail_eval_sampler, _ = make_tail_loader(
        cfg, train_loader.dataset, rank, world_size, shuffle=False)
    tail_iterator = iter(tail_loader)
    tail_cycle = 0

    model = get_model(
        cfg.network,
        dropout=0.0,
        fp16=False,
        num_features=cfg.embedding_size,
        herpn_range_limit=float(cfg.herpn_range_limit),
        herpn_bn_eps=float(cfg.herpn_bn_eps),
        herpn_progress=5.0,
        herpn_range_penalty_mode=str(cfg.herpn_range_penalty_mode),
        herpn_range_topk_fraction=float(getattr(
            cfg, "herpn_range_topk_fraction", 0.001)),
        herpn_range_bulk_weight=float(getattr(
            cfg, "herpn_range_bulk_weight", 0.01)),
        herpn_range_guard_ratio=float(cfg.herpn_range_guard_ratio),
        # Clean batches use the exact graph.  Stabilization is enabled only
        # around the separate tail forward below.
        herpn_training_stabilization_limit=None,
    )
    load_backbone_checkpoint(model, cfg.backbone_init)
    model.set_herpn_progress(5.0)
    if cfg.sync_bn:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.to(device)
    trainable_names, trainable_parameters = select_recovery_parameters(model)
    model = nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        broadcast_buffers=False,
        find_unused_parameters=False,
    )

    teacher = get_model(
        cfg.embedding_teacher_network,
        dropout=0.0,
        fp16=False,
        num_features=cfg.embedding_size,
    ).to(device)
    load_backbone_checkpoint(teacher, cfg.embedding_teacher_checkpoint)
    teacher.eval().requires_grad_(False)

    optimizer = torch.optim.SGD(
        trainable_parameters,
        lr=float(cfg.lr),
        momentum=0.0,
        weight_decay=0.0,
    )
    total_steps = len(train_loader) * int(cfg.num_epoch)
    scheduler = CosineLRWarmup(
        optimizer,
        warmup_iters=0,
        total_iters=total_steps,
        min_lr_ratio=float(cfg.min_lr_ratio),
    )
    anchors = tuple(
        parameter.detach().clone() for parameter in trainable_parameters)

    if rank == 0:
        logging.info(
            "Conflict-aware recovery starts from %s: ranks=%d, "
            "trainable_tensors=%d, tail_orientations=%d, epochs=%d",
            cfg.backbone_init, world_size, len(trainable_parameters),
            tail_count, cfg.num_epoch)
        logging.info(
            "Trainable scope: Conv + ordinary BN affine + HerPN weight/bias; "
            "FC/final embedding BN/dormant PReLU frozen")
        logging.info(
            "HerPN target=[-%g,%g], guard_ratio=%g, causal=%s",
            cfg.herpn_range_limit, cfg.herpn_range_limit,
            cfg.herpn_range_guard_ratio,
            tuple(cfg.tail_causal_activation_names))

    global_step = 0
    skipped_nonfinite = 0
    trust_stats = {"projected": 0, "worst_relative_delta": 0.0}
    for epoch in range(int(cfg.num_epoch)):
        train_loader.sampler.set_epoch(epoch)
        tail_eval_sampler.set_epoch(epoch)
        model.train()
        for step, batch in enumerate(train_loader):
            global_step += 1
            images = batch[0]
            if images.device != device:
                images = images.to(device=device, non_blocking=True)
            bn_snapshot = snapshot_batchnorm_buffers(model.module)

            # Clean forward: exact polynomial, train-mode BN, teacher geometry.
            model.module.set_herpn_training_stabilization(None)
            clean_embeddings = model.module(images)
            clean_is_finite = bool(torch.isfinite(clean_embeddings).all().item())
            if not all_ranks_true(clean_is_finite, device):
                skipped_nonfinite += 1
                restore_batchnorm_buffers(bn_snapshot)
                optimizer.zero_grad(set_to_none=True)
                if rank == 0:
                    logging.warning(
                        "Skipped non-finite clean batch at step %d (count=%d)",
                        global_step, skipped_nonfinite)
                continue
            with torch.no_grad():
                teacher_embeddings = teacher(images)
            clean_unit = F.normalize(clean_embeddings.float(), dim=1, eps=1e-6)
            teacher_unit = F.normalize(
                teacher_embeddings.float(), dim=1, eps=1e-6)
            clean_loss = float(cfg.clean_distill_weight) * 0.5 * (
                clean_unit - teacher_unit).square().sum(dim=1).mean()

            tail_loss = None
            tail_penalty_value = 0.0
            if global_step % int(cfg.fixed_tail_replay_interval) == 0:
                tail_batch, tail_iterator, tail_cycle = next_cycled(
                    tail_iterator, tail_loader, tail_sampler, tail_cycle)
                tail_images = tail_batch[0].to(
                    device=device, non_blocking=True)
                model.module.set_herpn_training_stabilization(
                    float(cfg.herpn_training_stabilization_limit),
                    tuple(cfg.herpn_training_stabilization_names),
                )
                # Tail forward: inference BN without updating its buffers;
                # activation modules remain in train mode to expose the loss.
                with temporary_batchnorm_eval(model.module):
                    tail_embeddings = model.module(tail_images)
                model.module.set_herpn_training_stabilization(None)
                tail_penalty = model.module.herpn_causal_range_penalty(
                    tuple(cfg.tail_causal_activation_names))
                tail_loss = float(cfg.tail_range_loss_weight) * tail_penalty
                tail_penalty_value = float(tail_penalty.detach().item())
                tail_is_finite = bool(
                    torch.isfinite(tail_embeddings).all().item()
                    and torch.isfinite(tail_loss).item())
                if not all_ranks_true(tail_is_finite, device):
                    skipped_nonfinite += 1
                    restore_batchnorm_buffers(bn_snapshot)
                    optimizer.zero_grad(set_to_none=True)
                    if rank == 0:
                        logging.warning(
                            "Skipped non-finite stabilized tail at step %d "
                            "(count=%d)", global_step, skipped_nonfinite)
                    continue

            clean_gradients = torch.autograd.grad(
                clean_loss,
                trainable_parameters,
                allow_unused=True,
            )
            if tail_loss is None:
                tail_gradients = (None,) * len(trainable_parameters)
            else:
                tail_gradients = torch.autograd.grad(
                    tail_loss,
                    trainable_parameters,
                    allow_unused=True,
                )
            finite_gradients = gradients_are_finite(
                clean_gradients, tail_gradients)
            if not all_ranks_true(finite_gradients, device):
                skipped_nonfinite += 1
                restore_batchnorm_buffers(bn_snapshot)
                optimizer.zero_grad(set_to_none=True)
                if rank == 0:
                    logging.warning(
                        "Skipped non-finite objective gradients at step %d "
                        "(count=%d)", global_step, skipped_nonfinite)
                continue

            current_lr = float(optimizer.param_groups[0]["lr"])
            combined, gradient_stats = combine_conflict_aware_gradients(
                trainable_parameters,
                clean_gradients,
                tail_gradients,
                learning_rate=current_lr,
                tail_to_clean_ratio=float(
                    cfg.tail_to_clean_gradient_ratio),
                max_step_update_ratio=float(cfg.max_step_update_ratio),
                scale_floor=float(cfg.parameter_scale_floor),
            )
            synchronize_and_assign_gradients(trainable_parameters, combined)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if global_step % int(cfg.parameter_trust_region_interval) == 0:
                trust_stats = project_to_relative_trust_region(
                    trainable_parameters,
                    anchors,
                    ratio=float(cfg.parameter_trust_region_ratio),
                    scale_floor=float(cfg.parameter_scale_floor),
                )
            scheduler.step()

            if rank == 0 and global_step % int(cfg.frequent) == 0:
                logging.info(
                    "Epoch %d step %d/%d global=%d lr=%.7g clean=%.7g "
                    "tail=%.7g conflicts=%d tail_limited=%d "
                    "update_limited=%d trust_projected=%d trust_max=%.7g "
                    "skips=%d",
                    epoch + 1, step + 1, len(train_loader), global_step,
                    current_lr, float(clean_loss.detach().item()),
                    tail_penalty_value, gradient_stats["conflicts"],
                    gradient_stats["tail_limited"],
                    gradient_stats["update_limited"],
                    trust_stats["projected"],
                    trust_stats["worst_relative_delta"],
                    skipped_nonfinite,
                )

        # Enforce the cumulative parameter boundary before every saved model.
        trust_stats = project_to_relative_trust_region(
            trainable_parameters,
            anchors,
            ratio=float(cfg.parameter_trust_region_ratio),
            scale_floor=float(cfg.parameter_scale_floor),
        )
        nonfinite_tail, evaluated_tail = exact_tail_gate(
            model.module, tail_eval_loader, device)
        model.train()
        if rank == 0:
            model_path = os.path.join(
                cfg.output, f"model_epoch_{epoch + 1:02d}.pt")
            atomic_torch_save(model.module.state_dict(), model_path)
            logging.info(
                "Epoch %d exact MS1Mv3 tail gate: nonfinite=%d/%d; saved %s",
                epoch + 1, nonfinite_tail, evaluated_tail, model_path)
        checkpoint = {
            "epoch": epoch + 1,
            "global_step": global_step,
            "state_dict_backbone": model.module.state_dict(),
            "state_optimizer": optimizer.state_dict(),
            "state_lr_scheduler": scheduler.state_dict(),
            "anchor_state": {
                name: anchor.cpu()
                for name, anchor in zip(trainable_names, anchors)
            },
            "skipped_nonfinite": skipped_nonfinite,
        }
        atomic_torch_save(
            checkpoint,
            os.path.join(cfg.output, f"checkpoint_gpu_{rank}.pt"),
        )
        distributed.barrier()

    if rank == 0:
        atomic_torch_save(
            model.module.state_dict(), os.path.join(cfg.output, "model.pt"))
        logging.info("Conflict-aware recovery complete")
    distributed.barrier()
    distributed.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    arguments = parser.parse_args()
    main(arguments.config)

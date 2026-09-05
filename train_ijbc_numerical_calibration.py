"""Iterative, BN-frozen numerical calibration of the full-poly R50 on IJB-C.

The saved inference graph remains the exact 25/25 degree-2 HerPN backbone.
Training-only straight-through bounds make catastrophic rows differentiable;
they are never present in evaluation or in a saved/FHE graph.
"""

import argparse
import json
import logging
import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributed
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

from backbones import get_model
from dataset import DatasetWithIndex, MXFaceDataset, PairedOrientationDataset
from mine_herpn_tails import atomic_json_dump
from train_tail_recovery import (
    atomic_torch_save,
    gradients_are_finite,
    load_backbone_checkpoint,
    select_recovery_parameters,
)
from utils.utils_config import get_config
from utils.utils_distributed_sampler import setup_seed
from utils.utils_ijbc_replay import (
    IJBCOrientationDataset,
    IJBCSourceDataset,
    load_ijbc_replay_orientations,
)
from utils.utils_logging import init_logging
from utils.utils_multi_objective import (
    combine_conflict_aware_gradients,
    project_to_relative_trust_region,
    synchronize_and_assign_gradients,
)


def freeze_batchnorm_buffers(model):
    count = 0
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()
            count += 1
    return count


def snapshot_batchnorm_buffers(model):
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
        if name.endswith(("running_mean", "running_var", "num_batches_tracked"))
    }


def assert_batchnorm_buffers_unchanged(model, reference):
    state = model.state_dict()
    changed = [name for name, value in reference.items()
               if not torch.equal(state[name].detach().cpu(), value)]
    if changed:
        raise RuntimeError(f"Frozen BatchNorm buffers changed: {changed[:5]}")


def make_loader(dataset, batch_size, workers, rank, world_size, shuffle, seed):
    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=shuffle,
        seed=int(seed), drop_last=False)
    loader = DataLoader(
        dataset, batch_size=int(batch_size), sampler=sampler,
        num_workers=int(workers), pin_memory=True, drop_last=False,
        persistent_workers=int(workers) > 0)
    return loader, sampler


def next_cycled(iterator, loader, sampler, cycle):
    try:
        return next(iterator), iterator, cycle
    except StopIteration:
        cycle += 1
        sampler.set_epoch(cycle)
        iterator = iter(loader)
        return next(iterator), iterator, cycle


def choose_preservation_orientations(source_count, excluded_sources, count, seed):
    candidates = [index for index in range(source_count)
                  if index not in excluded_sources]
    generator = random.Random(int(seed))
    generator.shuffle(candidates)
    source_limit = min(len(candidates), (int(count) + 1) // 2)
    rows = []
    for index in candidates[:source_limit]:
        rows.extend(((index, 0), (index, 1)))
    return tuple(rows[:int(count)])


@torch.no_grad()
def full_dataset_gate(model, loader, device, rank, world_size):
    model.set_herpn_training_stabilization(None)
    model.eval()
    local_failures = []
    for pairs, source_indices in loader:
        images = pairs.flatten(0, 1).to(device, non_blocking=True)
        embeddings = model(images)
        finite = torch.isfinite(embeddings).all(dim=1).cpu().tolist()
        indices = source_indices.repeat_interleave(2).tolist()
        orientations = [0, 1] * int(source_indices.numel())
        local_failures.extend(
            (int(index), orientation)
            for index, orientation, ok in zip(indices, orientations, finite)
            if not ok)
    gathered = [None] * world_size
    distributed.all_gather_object(gathered, local_failures)
    failures = sorted({row for shard in gathered for row in shard})
    return tuple(failures)


def build_model(cfg, checkpoint, device, penalty_mode=True):
    model = get_model(
        cfg.network, dropout=0.0, fp16=False,
        num_features=int(cfg.embedding_size),
        herpn_range_limit=float(cfg.herpn_range_limit),
        herpn_bn_eps=float(cfg.herpn_bn_eps), herpn_progress=5.0,
        herpn_range_penalty_mode=(
            str(cfg.herpn_range_penalty_mode) if penalty_mode else "legacy"),
        herpn_range_guard_ratio=float(cfg.herpn_range_guard_ratio),
    ).to(device)
    load_backbone_checkpoint(model, checkpoint)
    model.set_herpn_progress(5.0)
    return model


def make_calibration_datasets(cfg, local_rank):
    dataset_name = str(getattr(cfg, "calibration_dataset", "ijbc")).lower()
    if dataset_name == "ijbc":
        source_dataset = IJBCSourceDataset(cfg.ijbc_root, cfg.ijbc_target)

        def orientations(rows):
            return IJBCOrientationDataset(
                cfg.ijbc_root, rows, cfg.ijbc_target)

        return dataset_name, source_dataset, orientations
    if dataset_name == "ms1mv3":
        base_dataset = MXFaceDataset(cfg.rec, local_rank)
        source_dataset = PairedOrientationDataset(base_dataset)
        oriented_dataset = DatasetWithIndex(
            base_dataset, both_orientations=True)

        def orientations(rows):
            return Subset(
                oriented_dataset,
                [2 * int(index) + int(orientation)
                 for index, orientation in rows],
            )

        return dataset_name, source_dataset, orientations
    raise ValueError(
        "calibration_dataset must be either 'ijbc' or 'ms1mv3'")


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

    replay_manifest_paths = tuple(getattr(
        cfg, "calibration_replay_manifests", cfg.ijbc_replay_manifests))
    replay_activation_topk = int(getattr(
        cfg, "calibration_replay_activation_topk",
        cfg.ijbc_replay_activation_topk))
    replay = load_ijbc_replay_orientations(
        replay_manifest_paths, activation_topk=replay_activation_topk)
    priority_manifest_paths = tuple(getattr(
        cfg, "calibration_priority_manifests",
        getattr(cfg, "ijbc_priority_manifests", ())))
    priority = (
        load_ijbc_replay_orientations(priority_manifest_paths)
        if priority_manifest_paths else ())
    priority_repeats = int(getattr(cfg, "ijbc_gate_failure_repeats", 1))
    if priority_repeats <= 0:
        raise ValueError("ijbc_gate_failure_repeats must be positive")
    dataset_name, source_dataset, orientation_dataset = (
        make_calibration_datasets(cfg, local_rank))
    excluded_sources = {index for index, _ in replay}
    preservation = choose_preservation_orientations(
        len(source_dataset), excluded_sources,
        int(cfg.ijbc_preservation_count), int(cfg.seed))
    preservation_dataset = orientation_dataset(preservation)
    preservation_loader, preservation_sampler = make_loader(
        preservation_dataset, cfg.preservation_batch_size,
        cfg.ijbc_workers, rank, world_size, True, cfg.seed)
    preservation_iterator = iter(preservation_loader)
    preservation_cycle = 0
    gate_loader, gate_sampler = make_loader(
        source_dataset, cfg.full_gate_source_batch_size,
        cfg.ijbc_workers, rank, world_size, False, cfg.seed)

    model = build_model(cfg, cfg.backbone_init, device, penalty_mode=True)
    teacher = build_model(cfg, cfg.backbone_init, device, penalty_mode=False)
    teacher.eval().requires_grad_(False)
    trainable_names, trainable_parameters = select_recovery_parameters(model)
    optimizer = torch.optim.SGD(
        trainable_parameters, lr=float(cfg.lr), momentum=0.0,
        weight_decay=0.0)
    anchors = tuple(parameter.detach().clone()
                    for parameter in trainable_parameters)
    bn_reference = snapshot_batchnorm_buffers(model)
    activation_names = tuple(
        name for name, module in model.named_modules()
        if module.__class__.__name__ == "ProgressiveHerPNActivation")
    if len(activation_names) != 25:
        raise RuntimeError(f"Expected 25 HerPN activations, found {len(activation_names)}")
    frozen_batchnorm_count = freeze_batchnorm_buffers(model)

    if rank == 0:
        logging.info(
            "%s numerical calibration: checkpoint=%s, hard=%d, preserve=%d, "
            "trainable=%d, world=%d", dataset_name.upper(),
            cfg.backbone_init, len(replay),
            len(preservation), len(trainable_parameters), world_size)
        logging.info(
            "Exact target=PReLU on [-%g,%g], degree=2; training guard=%g; "
            "all %d BN modules (%d running buffers) are immutable",
            cfg.herpn_range_limit, cfg.herpn_range_limit,
            cfg.herpn_range_guard_ratio,
            frozen_batchnorm_count, len(bn_reference))

    global_step = 0
    skipped = 0
    background_replay = list(replay)
    replay_rows = list(priority) * priority_repeats + background_replay
    for epoch in range(1, int(cfg.num_epoch) + 1):
        preservation_sampler.set_epoch(epoch)
        preservation_iterator = iter(preservation_loader)
        preservation_cycle = epoch
        replay_dataset = orientation_dataset(replay_rows)
        replay_loader, replay_sampler = make_loader(
            replay_dataset, cfg.replay_batch_size, cfg.ijbc_workers,
            rank, world_size, True, int(cfg.seed) + epoch)
        replay_iterator = iter(replay_loader)
        replay_cycle = 0
        model.train()
        freeze_batchnorm_buffers(model)
        model.set_herpn_training_stabilization(
            float(cfg.herpn_training_stabilization_limit), activation_names)

        for step in range(1, int(cfg.steps_per_epoch) + 1):
            global_step += 1
            clean_batch, preservation_iterator, preservation_cycle = next_cycled(
                preservation_iterator, preservation_loader,
                preservation_sampler, preservation_cycle)
            tail_batch, replay_iterator, replay_cycle = next_cycled(
                replay_iterator, replay_loader, replay_sampler, replay_cycle)
            clean_images = clean_batch[0].to(device, non_blocking=True)
            tail_images = tail_batch[0].to(device, non_blocking=True)

            clean_embeddings = model(clean_images)
            with torch.no_grad():
                teacher_embeddings = teacher(clean_images)
            clean_loss = float(cfg.preservation_loss_weight) * 0.5 * (
                F.normalize(clean_embeddings.float(), dim=1, eps=1e-6)
                - F.normalize(teacher_embeddings.float(), dim=1, eps=1e-6)
            ).square().sum(dim=1).mean()

            tail_embeddings = model(tail_images)
            range_penalty = model.herpn_causal_range_penalty(
                activation_names, reduction=str(cfg.causal_range_reduction))
            tail_loss = float(cfg.tail_range_loss_weight) * range_penalty
            local_finite = bool(
                torch.isfinite(clean_embeddings).all().item()
                and torch.isfinite(tail_embeddings).all().item()
                and torch.isfinite(clean_loss).item()
                and torch.isfinite(tail_loss).item())
            finite_tensor = torch.tensor(
                int(local_finite), device=device, dtype=torch.long)
            distributed.all_reduce(finite_tensor, op=distributed.ReduceOp.MIN)
            if not finite_tensor.item():
                skipped += 1
                optimizer.zero_grad(set_to_none=True)
                continue

            clean_gradients = torch.autograd.grad(
                clean_loss, trainable_parameters, allow_unused=True)
            tail_gradients = torch.autograd.grad(
                tail_loss, trainable_parameters, allow_unused=True)
            local_finite = gradients_are_finite(clean_gradients, tail_gradients)
            finite_tensor.fill_(int(local_finite))
            distributed.all_reduce(finite_tensor, op=distributed.ReduceOp.MIN)
            if not finite_tensor.item():
                skipped += 1
                optimizer.zero_grad(set_to_none=True)
                continue
            combined, stats = combine_conflict_aware_gradients(
                trainable_parameters, clean_gradients, tail_gradients,
                learning_rate=float(cfg.lr),
                tail_to_clean_ratio=float(cfg.tail_to_clean_gradient_ratio),
                max_step_update_ratio=float(cfg.max_step_update_ratio),
                scale_floor=float(cfg.parameter_scale_floor))
            synchronize_and_assign_gradients(trainable_parameters, combined)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            trust = project_to_relative_trust_region(
                trainable_parameters, anchors,
                ratio=float(cfg.parameter_trust_region_ratio),
                scale_floor=float(cfg.parameter_scale_floor))
            if rank == 0 and step % int(cfg.frequent) == 0:
                logging.info(
                    "epoch=%d step=%d/%d clean=%.7g tail=%.7g conflicts=%d "
                    "tail_limited=%d update_limited=%d trust=%g skips=%d",
                    epoch, step, cfg.steps_per_epoch,
                    float(clean_loss.detach()), float(range_penalty.detach()),
                    stats["conflicts"], stats["tail_limited"],
                    stats["update_limited"], trust["worst_relative_delta"],
                    skipped)

        assert_batchnorm_buffers_unchanged(model, bn_reference)
        model_path = os.path.join(cfg.output, f"model_epoch_{epoch:02d}.pt")
        if rank == 0:
            atomic_torch_save(model.state_dict(), model_path)
        distributed.barrier()

        gate_sampler.set_epoch(epoch)
        failures = full_dataset_gate(
            model, gate_loader, device, rank, world_size)
        gate_payload = {
            "format": f"exact_{dataset_name}_numerical_gate_v1",
            "dataset": dataset_name,
            "epoch": epoch,
            "checkpoint": model_path,
            "total_augmented_embeddings": 2 * len(source_dataset),
            "nonfinite_count": len(failures),
            "output_nonfinite": [
                {"source_index": index, "orientation": orientation}
                for index, orientation in failures],
        }
        if rank == 0:
            atomic_json_dump(
                gate_payload,
                os.path.join(cfg.output, f"full_gate_epoch_{epoch:02d}.json"))
            logging.info(
                "Epoch %d full %s gate: nonfinite=%d/%d", epoch,
                dataset_name.upper(),
                len(failures), 2 * len(source_dataset))
        if not failures:
            if rank == 0:
                atomic_torch_save(
                    model.state_dict(),
                    os.path.join(cfg.output, "model_numerical_gate_zero.pt"))
                logging.info(
                    "Zero non-finite %s gate achieved; stopping",
                    dataset_name.upper())
            break
        known = set(background_replay)
        new_failures = [row for row in failures if row not in known]
        background_replay.extend(new_failures)
        # The active failures are deliberately duplicated.  Once only a few
        # rows remain, uniform replay of all previously resolved tails would
        # otherwise produce almost exclusively zero numerical losses.
        replay_rows = list(failures) * priority_repeats + background_replay
        if rank == 0:
            logging.info(
                "Added %d newly failing orientations; next replay=%d",
                len(new_failures), len(replay_rows))
        distributed.barrier()

    if rank == 0:
        atomic_torch_save(model.state_dict(), os.path.join(cfg.output, "model.pt"))
        with open(os.path.join(cfg.output, "calibration_state.json"), "w",
                  encoding="utf-8") as handle:
            json.dump({
                "global_step": global_step,
                "skipped_nonfinite_steps": skipped,
                "trainable_names": trainable_names,
                "final_replay_count": len(replay_rows),
            }, handle, indent=2)
            handle.write("\n")
    distributed.barrier()
    distributed.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    arguments = parser.parse_args()
    main(arguments.config)

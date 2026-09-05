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
def full_dataset_gate(
        model, loader, device, rank, world_size, *,
        activation_names=(), range_gate_limit=None):
    model.set_herpn_training_stabilization(None)
    model.eval()
    activation_names = tuple(activation_names)
    if range_gate_limit is not None and float(range_gate_limit) <= 0.0:
        raise ValueError("range_gate_limit must be positive or None")
    modules = dict(model.named_modules())
    unknown = sorted(set(activation_names).difference(modules))
    if unknown:
        raise ValueError(f"Unknown range-gate activations: {unknown}")
    local_output_failures = set()
    local_range_failures = {}
    batch_peaks = []
    handles = []
    if range_gate_limit is not None:
        def capture(_, inputs):
            values = inputs[0].detach().float()
            finite = torch.isfinite(values).flatten(1).all(dim=1)
            peaks = torch.nan_to_num(
                values.abs(), nan=torch.finfo(torch.float32).max,
                posinf=torch.finfo(torch.float32).max,
                neginf=torch.finfo(torch.float32).max,
            ).flatten(1).amax(dim=1)
            batch_peaks.append((peaks, finite))

        handles = [
            modules[name].register_forward_pre_hook(capture)
            for name in activation_names
        ]
    local_global_peak = torch.zeros((), device=device)
    try:
        for pairs, source_indices in loader:
            images = pairs.flatten(0, 1).to(device, non_blocking=True)
            indices = source_indices.repeat_interleave(2).tolist()
            orientations = [0, 1] * int(source_indices.numel())
            batch_peaks.clear()
            embeddings = model(images)
            finite_output = torch.isfinite(embeddings).all(dim=1).cpu().tolist()
            local_output_failures.update(
                (int(index), orientation)
                for index, orientation, ok in zip(
                    indices, orientations, finite_output)
                if not ok)
            if range_gate_limit is None:
                continue
            if len(batch_peaks) != len(activation_names):
                raise RuntimeError(
                    "Range gate did not observe every HerPN activation")
            peaks = torch.stack([row[0] for row in batch_peaks], dim=0)
            finite_inputs = torch.stack(
                [row[1] for row in batch_peaks], dim=0)
            violations = (~finite_inputs) | (peaks > float(range_gate_limit))
            any_violation = violations.any(dim=0)
            first_activation = violations.to(torch.uint8).argmax(dim=0)
            first_peak = peaks.gather(0, first_activation.unsqueeze(0)).squeeze(0)
            local_global_peak = torch.maximum(local_global_peak, peaks.amax())
            positions = any_violation.nonzero(as_tuple=False).flatten().cpu().tolist()
            first_activation = first_activation.cpu().tolist()
            first_peak = first_peak.cpu().tolist()
            for position in positions:
                key = (int(indices[position]), int(orientations[position]))
                local_range_failures.setdefault(key, (
                    activation_names[int(first_activation[position])],
                    float(first_peak[position]),
                ))
    finally:
        for handle in handles:
            handle.remove()
    gathered_outputs = [None] * world_size
    gathered_ranges = [None] * world_size
    distributed.all_gather_object(
        gathered_outputs, sorted(local_output_failures))
    distributed.all_gather_object(gathered_ranges, local_range_failures)
    distributed.all_reduce(local_global_peak, op=distributed.ReduceOp.MAX)
    output_failures = tuple(sorted({
        row for shard in gathered_outputs for row in shard
    }))
    range_failures = {}
    for shard in gathered_ranges:
        for key, value in shard.items():
            range_failures.setdefault(key, value)
    return {
        "output_nonfinite": output_failures,
        "range_violations": tuple(
            (key[0], key[1], value[0], value[1])
            for key, value in sorted(range_failures.items())
        ),
        "max_preherpn_absmax": (
            None if range_gate_limit is None
            else float(local_global_peak.cpu().item())
        ),
    }


def make_adversarial_tail_images(
        model, images, activation_names, *, epsilon, step_size, steps,
        random_start=True):
    """Maximize pre-HerPN ranges inside a bounded normalized-pixel ball."""
    epsilon = float(epsilon)
    step_size = float(step_size)
    steps = int(steps)
    if epsilon <= 0.0 or step_size <= 0.0 or steps <= 0:
        raise ValueError("adversarial epsilon/step-size/steps must be positive")
    clean = images.detach()
    lower = torch.maximum(clean - epsilon, clean.new_tensor(-1.0))
    upper = torch.minimum(clean + epsilon, clean.new_tensor(1.0))
    adversarial = clean.clone()
    if random_start:
        adversarial.add_(
            torch.empty_like(adversarial).uniform_(-epsilon, epsilon))
        adversarial = torch.maximum(torch.minimum(adversarial, upper), lower)
    last_objective = clean.new_zeros(())
    for _ in range(steps):
        adversarial.requires_grad_(True)
        model(adversarial)
        objective = model.herpn_adversarial_range_objective(
            activation_names, reduction="mean_max")
        gradient, = torch.autograd.grad(
            objective, adversarial, only_inputs=True)
        if not torch.isfinite(gradient).all():
            break
        with torch.no_grad():
            adversarial = adversarial + step_size * gradient.sign()
            adversarial = torch.maximum(
                torch.minimum(adversarial, upper), lower)
        last_objective = objective.detach()
    return adversarial.detach(), last_objective


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
    range_gate_limit = getattr(cfg, "numerical_range_gate_limit", None)
    range_gate_limit = (
        None if range_gate_limit is None else float(range_gate_limit))
    adversarial_enabled = bool(getattr(
        cfg, "adversarial_tail_enabled", False))

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
        logging.info(
            "Range gate=%s; adversarial tail=%s",
            "disabled" if range_gate_limit is None else range_gate_limit,
            adversarial_enabled)

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

            adversarial_objective = tail_images.new_zeros(())
            if adversarial_enabled:
                adversarial_images, adversarial_objective = (
                    make_adversarial_tail_images(
                        model, tail_images, activation_names,
                        epsilon=float(cfg.adversarial_tail_epsilon),
                        step_size=float(cfg.adversarial_tail_step_size),
                        steps=int(cfg.adversarial_tail_steps),
                        random_start=bool(cfg.adversarial_tail_random_start),
                    ))
                tail_images = torch.cat((tail_images, adversarial_images), dim=0)
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
                if adversarial_enabled:
                    logging.info(
                        "epoch=%d step=%d/%d adversarial_objective=%.7g",
                        epoch, step, cfg.steps_per_epoch,
                        float(adversarial_objective))

        assert_batchnorm_buffers_unchanged(model, bn_reference)
        model_path = os.path.join(cfg.output, f"model_epoch_{epoch:02d}.pt")
        if rank == 0:
            atomic_torch_save(model.state_dict(), model_path)
        distributed.barrier()

        gate_sampler.set_epoch(epoch)
        gate = full_dataset_gate(
            model, gate_loader, device, rank, world_size,
            activation_names=activation_names,
            range_gate_limit=range_gate_limit)
        output_failures = gate["output_nonfinite"]
        range_violations = gate["range_violations"]
        range_failure_rows = {
            (index, orientation)
            for index, orientation, _, _ in range_violations
        }
        failures = tuple(sorted(
            set(output_failures).union(range_failure_rows)))
        gate_payload = {
            "format": f"exact_{dataset_name}_numerical_gate_v1",
            "dataset": dataset_name,
            "epoch": epoch,
            "checkpoint": model_path,
            "total_augmented_embeddings": 2 * len(source_dataset),
            "nonfinite_count": len(output_failures),
            "range_gate_limit": range_gate_limit,
            "range_violation_count": len(range_violations),
            "numerical_failure_count": len(failures),
            "max_preherpn_absmax": gate["max_preherpn_absmax"],
            "output_nonfinite": [
                {"source_index": index, "orientation": orientation}
                for index, orientation in output_failures],
            "range_violations": [
                {
                    "source_index": index,
                    "orientation": orientation,
                    "first_activation": activation,
                    "input_absmax": peak,
                }
                for index, orientation, activation, peak in range_violations],
        }
        if rank == 0:
            atomic_json_dump(
                gate_payload,
                os.path.join(cfg.output, f"full_gate_epoch_{epoch:02d}.json"))
            logging.info(
                "Epoch %d full %s gate: nonfinite=%d range=%d "
                "numerical=%d/%d max_preherpn=%s", epoch,
                dataset_name.upper(), len(output_failures),
                len(range_violations), len(failures),
                2 * len(source_dataset), gate["max_preherpn_absmax"])
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

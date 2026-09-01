import argparse
import os
import time

import torch
from torch import distributed
from torch.utils.data import DataLoader

from backbones import get_model
from dataset import get_dataloader
from utils.utils_config import get_config
from utils.utils_distributed_sampler import setup_seed


def init_distributed():
    if "RANK" not in os.environ:
        return 0, 0, 1

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    distributed.init_process_group("nccl")
    return rank, local_rank, world_size


def begin_bn_recalibration(module, reset, momentum):
    """Keep the inference graph active while allowing BN statistics to update."""
    module.eval()
    batchnorm_layers = []
    for child in module.modules():
        if isinstance(child, torch.nn.modules.batchnorm._BatchNorm):
            if reset:
                child.reset_running_stats()
            child.momentum = momentum
            child.train()
            batchnorm_layers.append(child)
    if not batchnorm_layers:
        raise RuntimeError("Model contains no BatchNorm layers to recalibrate")
    return batchnorm_layers


def snapshot_bn_buffers(batchnorm_layers):
    """Copy all mutable BN buffers before a potentially rejected forward."""
    return [
        (
            layer,
            layer.running_mean.detach().clone(),
            layer.running_var.detach().clone(),
            layer.num_batches_tracked.detach().clone(),
        )
        for layer in batchnorm_layers
    ]


@torch.no_grad()
def restore_bn_buffers(snapshot):
    for layer, running_mean, running_var, num_batches_tracked in snapshot:
        layer.running_mean.copy_(running_mean)
        layer.running_var.copy_(running_var)
        layer.num_batches_tracked.copy_(num_batches_tracked)


def bn_buffers_are_finite(batchnorm_layers):
    return all(
        torch.isfinite(layer.running_mean).all()
        and torch.isfinite(layer.running_var).all()
        for layer in batchnorm_layers
    )


def synchronize_bad_forward(local_bad, device):
    bad = torch.tensor(int(bool(local_bad)), device=device, dtype=torch.int32)
    if distributed.is_available() and distributed.is_initialized():
        distributed.all_reduce(bad, op=distributed.ReduceOp.MAX)
    return bool(bad.item())


def merge_cumulative_bn_stats(batchnorm_layers):
    """Merge cumulative BN statistics from disjoint distributed data shards."""
    if not distributed.is_available() or not distributed.is_initialized():
        return

    for layer in batchnorm_layers:
        local_batches = layer.num_batches_tracked.detach().clone()
        total_batches = local_batches.clone()
        distributed.all_reduce(total_batches, op=distributed.ReduceOp.SUM)
        if total_batches.item() == 0:
            continue

        weight = local_batches.to(
            device=layer.running_mean.device,
            dtype=layer.running_mean.dtype,
        )
        for running_stat in (layer.running_mean, layer.running_var):
            weighted_stat = running_stat * weight
            distributed.all_reduce(weighted_stat, op=distributed.ReduceOp.SUM)
            running_stat.copy_(
                weighted_stat / total_batches.to(dtype=running_stat.dtype)
            )
        layer.num_batches_tracked.copy_(total_batches)


def parse_blends(value):
    try:
        blends = tuple(float(item.strip()) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "blends must be comma-separated numbers"
        ) from error
    if not blends or any(blend < 0.0 or blend > 1.0 for blend in blends):
        raise argparse.ArgumentTypeError(
            "blends must contain values in the closed interval [0, 1]"
        )
    return blends


def main(args):
    cfg = get_config(args.config)
    setup_seed(seed=cfg.seed, cuda_deterministic=False)

    rank, local_rank, world_size = init_distributed()
    if world_size > 1 and args.bn_momentum is not None:
        raise ValueError(
            "Distributed merging requires --bn-momentum=None so each rank "
            "produces cumulative statistics over its disjoint data shard"
        )
    torch.cuda.set_device(local_rank)

    model_kwargs = {
        "dropout": 0.0,
        "fp16": cfg.fp16,
        "num_features": cfg.embedding_size,
    }
    for config_name in ("herpn_range_limit", "herpn_bn_eps"):
        if hasattr(cfg, config_name):
            model_kwargs[config_name] = getattr(cfg, config_name)
    if cfg.network.startswith("poolformer_no_ln_x2_act"):
        model_kwargs["gate_grouping"] = str(getattr(
            cfg, "simple_gate_grouping", "stage_chunks"))
    if cfg.network == "patch_cnn":
        model_kwargs.update(
            input_size=getattr(cfg, "input_size", 112),
            patch_size=getattr(cfg, "patch_size", 28),
        )

    backbone = get_model(cfg.network, **model_kwargs).cuda()
    checkpoint = torch.load(args.model, map_location="cpu")
    checkpoint_blends = None
    if (isinstance(checkpoint, dict)
            and isinstance(checkpoint.get("state_dict_backbone"), dict)):
        checkpoint_blends = checkpoint.get("simple_gate_blends")
        state = checkpoint["state_dict_backbone"]
    else:
        state = checkpoint
    backbone.load_state_dict(state, strict=True)

    simple_gate_blends = (
        args.simple_gate_blends
        if args.simple_gate_blends is not None
        else checkpoint_blends
    )
    if simple_gate_blends is not None:
        if not hasattr(backbone, "set_simple_gate_blends"):
            raise ValueError(
                "SimpleGate blends were provided for a model without "
                "SimpleGate scheduling"
            )
        backbone.set_simple_gate_blends(simple_gate_blends)

    train_loader = get_dataloader(
        cfg.rec,
        local_rank,
        args.batch_size or cfg.batch_size,
        cfg.dali,
        cfg.dali_aug,
        cfg.seed,
        cfg.num_workers,
    )

    for param in backbone.parameters():
        param.requires_grad_(False)
    batchnorm_layers = begin_bn_recalibration(
        backbone,
        reset=args.reset_bn,
        momentum=args.bn_momentum,
    )

    attempted_batches = 0
    accepted_batches = 0
    skipped_batches = 0
    started_at = time.monotonic()
    with torch.no_grad():
        for epoch in range(args.epochs):
            if isinstance(train_loader, DataLoader):
                train_loader.sampler.set_epoch(epoch)
            for img, _ in train_loader:
                attempted_batches += 1
                snapshot = (
                    snapshot_bn_buffers(batchnorm_layers)
                    if args.skip_nonfinite else None
                )
                out = backbone(img)
                local_bad = (
                    not torch.isfinite(out).all()
                    or not bn_buffers_are_finite(batchnorm_layers)
                )
                bad = synchronize_bad_forward(local_bad, out.device)
                if bad:
                    if not args.skip_nonfinite:
                        raise FloatingPointError(
                            "Non-finite embedding or BatchNorm buffer while "
                            "refreshing statistics at "
                            f"epoch={epoch}, batch={attempted_batches - 1}"
                        )
                    restore_bn_buffers(snapshot)
                    skipped_batches += 1
                    if skipped_batches > args.max_nonfinite_skips:
                        raise FloatingPointError(
                            "BN recalibration exceeded the non-finite skip "
                            f"limit ({args.max_nonfinite_skips})"
                        )
                    if rank == 0:
                        print(
                            "Skipped synchronized non-finite BN calibration "
                            f"batch {attempted_batches - 1}; "
                            f"skip_count={skipped_batches}",
                            flush=True,
                        )
                    if (args.max_batches > 0
                            and attempted_batches >= args.max_batches):
                        break
                    continue
                accepted_batches += 1
                if (
                    rank == 0
                    and args.log_interval > 0
                    and attempted_batches % args.log_interval == 0
                ):
                    elapsed = time.monotonic() - started_at
                    global_batches = accepted_batches * world_size
                    print(
                        f"Accepted {global_batches} rank-local batches "
                        f"({global_batches * img.shape[0]} images) in "
                        f"{elapsed:.1f}s; synchronized_skips={skipped_batches}",
                        flush=True,
                    )
                if (args.max_batches > 0
                        and attempted_batches >= args.max_batches):
                    break
            if cfg.dali:
                train_loader.reset()
            if (args.max_batches > 0
                    and attempted_batches >= args.max_batches):
                break

    if accepted_batches == 0:
        raise RuntimeError("BN recalibration accepted no finite batches")

    merge_cumulative_bn_stats(batchnorm_layers)

    if rank == 0:
        output = args.output or args.model.replace(".pt", "_bnrefreshed.pt")
        output_directory = os.path.dirname(output)
        if output_directory:
            os.makedirs(output_directory, exist_ok=True)
        backbone.eval()
        torch.save(backbone.state_dict(), output)
        print(f"Saved BN-refreshed model to {output}")
        print(
            f"Used {accepted_batches * world_size} rank-local batches across "
            f"{world_size} rank(s); skipped {skipped_batches} synchronized "
            f"batch(es) out of {attempted_batches} attempts per rank"
        )

    if distributed.is_available() and distributed.is_initialized():
        distributed.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Refresh BatchNorm running stats for a saved backbone model.pt."
    )
    parser.add_argument("config", type=str, help="config path, e.g. configs/ms1mv3_r50_no_relu")
    parser.add_argument("--model", required=True, help="input model.pt")
    parser.add_argument("--output", default=None, help="output model path")
    parser.add_argument("--epochs", type=int, default=1, help="number of passes over training data")
    parser.add_argument("--batch-size", type=int, default=None, help="override config batch size")
    parser.add_argument("--max-batches", type=int, default=-1, help="limit batches; <=0 means full pass")
    parser.add_argument(
        "--bn-momentum",
        type=float,
        default=None,
        help="BN momentum used only during refresh; default None uses cumulative stats",
    )
    parser.add_argument(
        "--reset-bn",
        action="store_true",
        help="reset BN running_mean/running_var before recalibration",
    )
    parser.add_argument(
        "--skip-nonfinite",
        action="store_true",
        help=(
            "synchronously reject a non-finite forward and roll back every "
            "BatchNorm buffer instead of aborting"
        ),
    )
    parser.add_argument(
        "--max-nonfinite-skips",
        type=int,
        default=0,
        help="maximum synchronized rejected batches when --skip-nonfinite is set",
    )
    parser.add_argument(
        "--simple-gate-blends",
        type=parse_blends,
        default=None,
        metavar="B0,B1,...",
        help="explicit SimpleGate group blends used during recalibration",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=500,
        help="rank-local batch interval for progress logging; <=0 disables",
    )
    main(parser.parse_args())

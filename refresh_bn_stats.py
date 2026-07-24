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

    batches_seen = 0
    started_at = time.monotonic()
    with torch.no_grad():
        for epoch in range(args.epochs):
            if isinstance(train_loader, DataLoader):
                train_loader.sampler.set_epoch(epoch)
            for img, _ in train_loader:
                out = backbone(img)
                if not torch.isfinite(out).all():
                    raise FloatingPointError(
                        "Non-finite embedding while refreshing BN stats "
                        f"at epoch={epoch}, batch={batches_seen}"
                    )
                batches_seen += 1
                if (
                    rank == 0
                    and args.log_interval > 0
                    and batches_seen % args.log_interval == 0
                ):
                    elapsed = time.monotonic() - started_at
                    global_batches = batches_seen * world_size
                    print(
                        f"Recalibrated {global_batches} global batches "
                        f"({global_batches * img.shape[0]} images) in "
                        f"{elapsed:.1f}s",
                        flush=True,
                    )
                if args.max_batches > 0 and batches_seen >= args.max_batches:
                    break
            if cfg.dali:
                train_loader.reset()
            if args.max_batches > 0 and batches_seen >= args.max_batches:
                break

    merge_cumulative_bn_stats(batchnorm_layers)
    total_batches = torch.tensor(
        batches_seen,
        dtype=torch.long,
        device=torch.device("cuda", local_rank),
    )
    if world_size > 1:
        distributed.all_reduce(total_batches, op=distributed.ReduceOp.SUM)

    if rank == 0:
        output = args.output or args.model.replace(".pt", "_bnrefreshed.pt")
        backbone.eval()
        torch.save(backbone.state_dict(), output)
        print(f"Saved BN-refreshed model to {output}")
        print(f"Used {total_batches.item()} batches across {world_size} rank(s)")

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

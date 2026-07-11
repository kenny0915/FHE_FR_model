import argparse
import os

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


def reset_bn_running_stats(module):
    for child in module.modules():
        if isinstance(child, torch.nn.modules.batchnorm._BatchNorm):
            child.reset_running_stats()


def set_bn_momentum(module, momentum):
    for child in module.modules():
        if isinstance(child, torch.nn.modules.batchnorm._BatchNorm):
            child.momentum = momentum


def main(args):
    cfg = get_config(args.config)
    setup_seed(seed=cfg.seed, cuda_deterministic=False)

    rank, local_rank, world_size = init_distributed()
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
    state = torch.load(args.model, map_location="cpu")
    backbone.load_state_dict(state, strict=True)

    if args.reset_bn:
        reset_bn_running_stats(backbone)
    set_bn_momentum(backbone, args.bn_momentum)

    train_loader = get_dataloader(
        cfg.rec,
        local_rank,
        args.batch_size or cfg.batch_size,
        cfg.dali,
        cfg.dali_aug,
        cfg.seed,
        cfg.num_workers,
    )

    backbone.train()
    for param in backbone.parameters():
        param.requires_grad_(False)

    batches_seen = 0
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
                if args.max_batches > 0 and batches_seen >= args.max_batches:
                    break
            if cfg.dali:
                train_loader.reset()
            if args.max_batches > 0 and batches_seen >= args.max_batches:
                break

    if world_size > 1:
        distributed.barrier()

    if rank == 0:
        output = args.output or args.model.replace(".pt", "_bnrefreshed.pt")
        backbone.eval()
        torch.save(backbone.state_dict(), output)
        print(f"Saved BN-refreshed model to {output}")
        print(f"Used {batches_seen} batches")

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
    main(parser.parse_args())

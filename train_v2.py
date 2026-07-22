import argparse
import json
import logging
import math
import os
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from backbones import get_model
from dataset import get_dataloader
from losses import build_margin_loss
from lr_scheduler import PolynomialLRWarmup
from partial_fc_v2 import PartialFC_V2
from torch import distributed
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils.utils_callbacks import CallBackLogging, CallBackVerification
from utils.utils_config import get_config
from utils.utils_distributed_sampler import setup_seed
from utils.utils_logging import AverageMeter, init_logging
from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import fp16_compress_hook

assert torch.__version__ >= "1.12.0", "In order to enjoy the features of the new torch, \
we have upgraded the torch to 1.12.0. torch before than 1.12.0 may not work in the future."

try:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    distributed.init_process_group("nccl")
except KeyError:
    rank = 0
    local_rank = 0
    world_size = 1
    distributed.init_process_group(
        backend="nccl",
        init_method="tcp://127.0.0.1:12584",
        rank=rank,
        world_size=world_size,
    )


def check_finite_gradients(module: torch.nn.Module, name: str, global_step: int):
    for param_name, param in module.named_parameters():
        if param.grad is not None and not torch.isfinite(param.grad).all():
            grad = param.grad.detach()
            finite = grad[torch.isfinite(grad)]
            finite_max = finite.abs().max().item() if finite.numel() > 0 else float('nan')
            raise FloatingPointError(
                f"Non-finite gradient in {name}.{param_name} at global_step={global_step}; "
                f"shape={tuple(grad.shape)}, finite_abs_max={finite_max}"
            )


def set_prepbn_progress(module: torch.nn.Module, current_step: int, total_steps: int):
    for submodule in module.modules():
        if hasattr(submodule, "set_progress"):
            submodule.set_progress(current_step, total_steps)


def prepbn_transition_complete(current_step: int, total_steps: int) -> bool:
    return total_steps <= 0 or current_step >= total_steps


def simple_gate_blends_at_epoch(epoch_value, group_epochs, transition_epochs):
    """Return one GELU-to-gate blend for each ordered conversion group."""
    starts = tuple(float(value) for value in group_epochs)
    transition_epochs = float(transition_epochs)
    if not starts:
        return ()
    if transition_epochs <= 0:
        raise ValueError("simple_gate_transition_epochs must be positive")
    if any(right < left + transition_epochs
           for left, right in zip(starts, starts[1:])):
        raise ValueError(
            "SimpleGate transitions must be ordered and non-overlapping")
    return tuple(
        min(max((float(epoch_value) - start) / transition_epochs, 0.0), 1.0)
        for start in starts
    )


def set_simple_gate_instrumentation(module: torch.nn.Module, enabled: bool,
                                    gradient_scale: float = 1.0):
    setter = getattr(module, "set_simple_gate_instrumentation", None)
    if setter is not None:
        setter(enabled, gradient_scale=gradient_scale)


def collect_simple_gate_stats(module: torch.nn.Module):
    collector = getattr(module, "simple_gate_range_stats", None)
    return collector() if collector is not None else {}


def set_simple_gate_blends(module: torch.nn.Module, blends):
    setter = getattr(module, "set_simple_gate_blends", None)
    if setter is not None:
        setter(blends)


def serialize_simple_gate_stats(stats):
    return {
        layer_name: {
            metric: float(value.detach().item())
            for metric, value in layer_stats.items()
        }
        for layer_name, layer_stats in stats.items()
    }


def log_simple_gate_stats(stats, global_step, summary_writer=None,
                          wandb_logger=None, prefix="SimpleGate"):
    """Log every gate plus compact network-wide stability summaries."""
    if not stats:
        return {}
    serialized = serialize_simple_gate_stats(stats)
    for layer_name, layer_stats in serialized.items():
        tensorboard_layer = layer_name.replace(".", "/")
        if summary_writer is not None:
            for metric, value in layer_stats.items():
                summary_writer.add_scalar(
                    f"{prefix}/{tensorboard_layer}/{metric}", value, global_step)

    product_absmax = max(
        values["product_absmax"] for values in serialized.values())
    product_p999 = max(
        values["product_p999"] for values in serialized.values())
    outside_fraction = sum(
        values["product_outside_fraction"] for values in serialized.values()
    ) / len(serialized)
    gradient_absmax = max(
        (values.get("gradient_absmax", 0.0) for values in serialized.values()),
        default=0.0,
    )
    summary = {
        "product_absmax": product_absmax,
        "product_p999": product_p999,
        "product_outside_fraction": outside_fraction,
        "gradient_absmax": gradient_absmax,
    }
    if summary_writer is not None:
        for metric, value in summary.items():
            summary_writer.add_scalar(f"{prefix}/Summary/{metric}", value, global_step)
    if wandb_logger:
        wandb_logger.log({
            f"{prefix}/Summary/{metric}": value for metric, value in summary.items()
        })

    worst_layers = sorted(
        serialized.items(),
        key=lambda item: item[1]["product_absmax"],
        reverse=True,
    )[:3]
    logging.info(
        "[%s][%d] product_absmax=%.6g p99.9=%.6g outside=%.6g "
        "gradient_absmax=%.6g worst=%s",
        prefix,
        global_step,
        product_absmax,
        product_p999,
        outside_fraction,
        gradient_absmax,
        ", ".join(
            f"{name}:{values['product_absmax']:.6g}"
            for name, values in worst_layers
        ),
    )
    return serialized


def begin_batchnorm_recalibration(module: torch.nn.Module, reset=True):
    """Use the final eval graph while updating only BatchNorm statistics."""
    batchnorm_state = [
        (submodule, submodule.training, submodule.momentum)
        for submodule in module.modules()
        if isinstance(submodule, nn.modules.batchnorm._BatchNorm)
    ]
    state = {
        "model_training": module.training,
        "batchnorm": batchnorm_state,
    }
    module.eval()
    for submodule, _, _ in batchnorm_state:
        if reset:
            submodule.reset_running_stats()
        submodule.momentum = None
        submodule.train()
    return state


def end_batchnorm_recalibration(module: torch.nn.Module, state):
    module.train(state["model_training"])
    for submodule, was_training, momentum in state["batchnorm"]:
        submodule.momentum = momentum
        submodule.train(was_training)


@torch.no_grad()
def recalibrate_prepbn_batchnorm(backbone, train_loader, num_epochs, start_epoch,
                                dali=False):
    """Reset and refresh BN statistics through the fully converted RepBN path."""
    if num_epochs <= 0:
        return 0
    module = backbone.module
    state = begin_batchnorm_recalibration(module, reset=True)
    completed = 0
    try:
        for stat_epoch in range(num_epochs):
            if isinstance(train_loader, DataLoader):
                sampler = train_loader.sampler
                if hasattr(sampler, "set_epoch"):
                    sampler.set_epoch(start_epoch + stat_epoch)
            for img, _ in train_loader:
                embeddings = backbone(img)
                if not torch.isfinite(embeddings).all():
                    raise FloatingPointError(
                        "Non-finite embeddings during final RepBatchNorm recalibration")
                completed += 1
            if dali:
                train_loader.reset()
    finally:
        end_batchnorm_recalibration(module, state)
    if completed == 0:
        raise RuntimeError("RepBatchNorm recalibration received no batches")
    return completed


@torch.no_grad()
def profile_simple_gate_ranges(backbone, train_loader, num_batches, dali=False):
    """Profile the final eval/RepBN graph over representative training images."""
    module = backbone.module
    if num_batches <= 0 or not hasattr(module, "simple_gate_range_stats"):
        return {}
    was_training = module.training
    module.eval()
    accumulated = {}
    completed = 0
    try:
        for img, _ in train_loader:
            set_simple_gate_instrumentation(module, True)
            embeddings = backbone(img)
            if not torch.isfinite(embeddings).all():
                raise FloatingPointError(
                    "Non-finite embeddings during final SimpleGate profiling")
            batch_stats = serialize_simple_gate_stats(
                collect_simple_gate_stats(module))
            for layer_name, layer_stats in batch_stats.items():
                target = accumulated.setdefault(
                    layer_name,
                    {metric: [] for metric in layer_stats},
                )
                for metric, value in layer_stats.items():
                    target.setdefault(metric, []).append(value)
            completed += 1
            if completed >= num_batches:
                break
    finally:
        set_simple_gate_instrumentation(module, False)
        module.train(was_training)
        if dali:
            train_loader.reset()
    if completed == 0:
        raise RuntimeError("Final SimpleGate profiling received no batches")

    max_metrics = {
        "operand1_absmax", "operand2_absmax", "product_absmax",
    }
    min_metrics = {"finite", "gradient_finite"}
    result = {}
    profile_device = next(module.parameters()).device
    for layer_name, layer_stats in accumulated.items():
        result[layer_name] = {}
        for metric, values in layer_stats.items():
            if metric in max_metrics:
                reduced = max(values)
            elif metric in min_metrics:
                reduced = min(values)
            else:
                reduced = sum(values) / len(values)
            if distributed.is_initialized() and distributed.get_world_size() > 1:
                reduced_tensor = torch.tensor(
                    reduced, device=profile_device)
                if metric in max_metrics:
                    op = distributed.ReduceOp.MAX
                elif metric in min_metrics:
                    op = distributed.ReduceOp.MIN
                else:
                    op = distributed.ReduceOp.SUM
                distributed.all_reduce(reduced_tensor, op=op)
                reduced = float(reduced_tensor.item())
                if op == distributed.ReduceOp.SUM:
                    reduced /= distributed.get_world_size()
            result[layer_name][metric] = reduced
    total_profile_batches = completed
    if distributed.is_initialized() and distributed.get_world_size() > 1:
        completed_tensor = torch.tensor(
            completed, device=profile_device, dtype=torch.long)
        distributed.all_reduce(completed_tensor, op=distributed.ReduceOp.SUM)
        total_profile_batches = int(completed_tensor.item())
    result["_profile"] = {
        "num_batches_across_ranks": total_profile_batches,
        "absmax_reduction": "maximum across batches and ranks",
        "other_metric_reduction": "mean across batches and ranks",
    }
    return result


def herpn_progress_at_epoch(epoch_value, stage_epochs, transition_epochs):
    if not stage_epochs:
        return 5.0
    transition_epochs = float(transition_epochs)
    if transition_epochs <= 0:
        raise ValueError("herpn_transition_epochs must be positive")
    starts = tuple(float(value) for value in stage_epochs)
    if len(starts) != 5:
        raise ValueError("herpn_stage_epochs must contain stem/layer1/layer2/layer3/layer4 starts")
    if any(right < left + transition_epochs for left, right in zip(starts, starts[1:])):
        raise ValueError("HerPN stage transitions must be ordered and non-overlapping")
    return sum(min(max((float(epoch_value) - start) / transition_epochs, 0.0), 1.0)
               for start in starts)


def herpn_group_blends_at_epoch(epoch_value, conversion_groups, group_epochs,
                                transition_epochs):
    groups = tuple(tuple(group) for group in conversion_groups)
    starts = tuple(float(value) for value in group_epochs)
    transition_epochs = float(transition_epochs)
    if len(groups) != len(starts):
        raise ValueError(
            "herpn_conversion_groups and herpn_group_epochs must have equal length")
    if transition_epochs <= 0:
        raise ValueError("herpn_transition_epochs must be positive")
    if any(right < left + transition_epochs
           for left, right in zip(starts, starts[1:])):
        raise ValueError("HerPN conversion groups must be ordered and non-overlapping")

    blends = {}
    for group, start in zip(groups, starts):
        blend = min(max(
            (float(epoch_value) - start) / transition_epochs, 0.0), 1.0)
        for activation_name in group:
            if activation_name in blends:
                raise ValueError(
                    f"HerPN activation appears in multiple groups: {activation_name}")
            blends[activation_name] = blend
    return blends


def validate_herpn_conversion_groups(module, conversion_groups):
    expected = {
        name for name, submodule in module.named_modules()
        if submodule.__class__.__name__ == "ProgressiveHerPNActivation"
    }
    scheduled = {name for group in conversion_groups for name in group}
    count = sum(len(group) for group in conversion_groups)
    if count != len(scheduled):
        raise ValueError("An activation occurs more than once in HerPN groups")
    missing = sorted(expected.difference(scheduled))
    unknown = sorted(scheduled.difference(expected))
    if missing or unknown:
        raise ValueError(
            f"Invalid HerPN conversion groups; missing={missing}, unknown={unknown}")


def atomic_torch_save(value, path):
    """Write a checkpoint completely before replacing an existing file."""
    temporary_path = path + ".tmp"
    torch.save(value, temporary_path)
    os.replace(temporary_path, path)


@torch.no_grad()
def recalibrate_herpn_batchnorm(backbone, train_loader, num_batches, global_step):
    module = backbone.module
    if num_batches <= 0 or not hasattr(module, "begin_batchnorm_recalibration"):
        return
    state = module.begin_batchnorm_recalibration(reset=True)
    completed = 0
    try:
        for batch in train_loader:
            img = batch[0]
            embeddings = backbone(img)
            if not torch.isfinite(embeddings).all():
                raise FloatingPointError(
                    "Non-finite embeddings during HerPN BatchNorm recalibration "
                    f"at global_step={global_step}, calibration_batch={completed}"
                )
            completed += 1
            if completed >= num_batches:
                break
    finally:
        module.end_batchnorm_recalibration(state)
    if completed == 0:
        raise RuntimeError("HerPN BatchNorm recalibration received no batches")


@torch.no_grad()
def recalibrate_batchnorm_batches(backbone, train_loader, num_batches,
                                  global_step, reason):
    """Reset and refresh BN statistics using the current inference graph."""
    if num_batches <= 0:
        return 0
    module = backbone.module
    state = begin_batchnorm_recalibration(module, reset=True)
    completed = 0
    try:
        for img, _ in train_loader:
            embeddings = backbone(img)
            if not torch.isfinite(embeddings).all():
                raise FloatingPointError(
                    f"Non-finite embeddings during {reason} BatchNorm "
                    f"recalibration at global_step={global_step}, "
                    f"calibration_batch={completed}")
            completed += 1
            if completed >= num_batches:
                break
    finally:
        end_batchnorm_recalibration(module, state)
    if completed == 0:
        raise RuntimeError(f"{reason} BatchNorm recalibration received no batches")
    return completed



class CryptoFaceArcFaceHead(nn.Module):
    def __init__(self, embedding_dim: int, num_classes: int, s: float = 64.0, m: float = 0.5):
        super().__init__()
        self.kernel = nn.Parameter(torch.empty(embedding_dim, num_classes))
        self.kernel.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        self.s = s
        self.m = m
        self.eps = 1e-4

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor):
        embeddings = embeddings / torch.norm(embeddings, 2, 1, True)
        kernel = self.kernel / torch.norm(self.kernel, 2, 0, True)
        cosine = (embeddings @ kernel).clamp(-1 + self.eps, 1 - self.eps)

        m_hot = torch.zeros(labels.size(0), cosine.size(1), device=cosine.device, dtype=cosine.dtype)
        m_hot.scatter_(1, labels.reshape(-1, 1), self.m)

        theta = cosine.acos()
        theta_m = torch.clip(theta + m_hot, min=self.eps, max=math.pi - self.eps)
        return theta_m.cos() * self.s


def is_cryptoface_patch_training(cfg):
    return cfg.network == "patch_cnn" and getattr(cfg, "patch_cnn_training", "") == "cryptoface"

def main(args):

    # get config
    cfg = get_config(args.config)
    # global control random seed
    setup_seed(seed=cfg.seed, cuda_deterministic=False)

    torch.cuda.set_device(local_rank)

    os.makedirs(cfg.output, exist_ok=True)
    init_logging(rank, cfg.output)

    summary_writer = (
        SummaryWriter(log_dir=os.path.join(cfg.output, "tensorboard"))
        if rank == 0
        else None
    )
    
    wandb_logger = None
    if cfg.using_wandb:
        import wandb
        # Sign in to wandb
        try:
            wandb.login(key=cfg.wandb_key)
        except Exception as e:
            print("WandB Key must be provided in config file (base.py).")
            print(f"Config Error: {e}")
        # Initialize wandb
        run_name = datetime.now().strftime("%y%m%d_%H%M") + f"_GPU{rank}"
        run_name = run_name if cfg.suffix_run_name is None else run_name + f"_{cfg.suffix_run_name}"
        try:
            wandb_logger = wandb.init(
                entity = cfg.wandb_entity, 
                project = cfg.wandb_project, 
                sync_tensorboard = True,
                resume=cfg.wandb_resume,
                name = run_name, 
                notes = cfg.notes) if rank == 0 or cfg.wandb_log_all else None
            if wandb_logger:
                wandb_logger.config.update(cfg)
        except Exception as e:
            print("WandB Data (Entity and Project name) must be provided in config file (base.py).")
            print(f"Config Error: {e}")
    train_loader = get_dataloader(
        cfg.rec,
        local_rank,
        cfg.batch_size,
        cfg.dali,
        cfg.dali_aug,
        cfg.seed,
        cfg.num_workers
    )

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
    if cfg.network.startswith("r") and cfg.network.endswith("_no_relu"):
        default_herpn_progress = (
            0.0 if (getattr(cfg, "herpn_conversion_groups", ())
                    or getattr(cfg, "herpn_stage_epochs", ())) else 5.0)
        model_kwargs.update(
            herpn_range_limit=float(getattr(cfg, "herpn_range_limit", 6.0)),
            herpn_bn_eps=float(getattr(cfg, "herpn_bn_eps", 1e-4)),
            herpn_progress=float(getattr(
                cfg, "herpn_initial_progress", default_herpn_progress)),
        )
    if cfg.network.startswith("poolformer_no_ln_x2_act"):
        gate_group_epochs = tuple(getattr(
            cfg, "simple_gate_group_epochs", ()))
        model_kwargs.update(
            gate_range_limit=float(getattr(cfg, "simple_gate_range_limit", 6.0)),
            gate_stats_sample_size=int(getattr(
                cfg, "simple_gate_stats_sample_size", 16384)),
            gate_compute_fp32=bool(getattr(
                cfg, "simple_gate_compute_fp32", True)),
            gate_fail_on_nonfinite=bool(getattr(
                cfg, "simple_gate_fail_on_nonfinite", True)),
            gate_initial_blend=float(getattr(
                cfg, "simple_gate_initial_blend",
                0.0 if gate_group_epochs else 1.0)),
        )

    backbone = get_model(cfg.network, **model_kwargs).cuda()
    backbone_init = getattr(cfg, "backbone_init", "")
    if backbone_init and not cfg.resume:
        init_checkpoint = torch.load(backbone_init, map_location="cpu")
        if "state_dict_backbone" in init_checkpoint:
            init_checkpoint = init_checkpoint["state_dict_backbone"]
        backbone.load_state_dict(init_checkpoint, strict=True)
        if hasattr(backbone, "set_herpn_progress"):
            backbone.set_herpn_progress(
                float(getattr(cfg, "herpn_initial_progress", 0.0)))
        logging.info("Initialized backbone from %s", backbone_init)
        del init_checkpoint
    if getattr(cfg, "sync_bn", False):
        backbone = torch.nn.SyncBatchNorm.convert_sync_batchnorm(backbone)

    backbone = torch.nn.parallel.DistributedDataParallel(
        module=backbone,
        broadcast_buffers=bool(getattr(cfg, "broadcast_buffers", True)),
        device_ids=[local_rank], bucket_cap_mb=16,
        find_unused_parameters=True)
    backbone.register_comm_hook(None, fp16_compress_hook)

    backbone.train()
    # FIXME using gradient checkpoint if there are some unused parameters will cause error
    backbone._set_static_graph()

    cryptoface_patch_training = is_cryptoface_patch_training(cfg)
    if cryptoface_patch_training and world_size != 1:
        raise ValueError("patch_cnn_training='cryptoface' matches the CryptoFace single-GPU training loop; use one process.")

    margin_loss = build_margin_loss(cfg)

    if cryptoface_patch_training:
        module_partial_fc = CryptoFaceArcFaceHead(
            cfg.embedding_size,
            cfg.num_classes,
            s=float(getattr(cfg, "scale", 64.0)),
            m=float(getattr(cfg, "cryptoface_arcface_margin", cfg.margin_list[1])),
        ).cuda()
        criterion = nn.CrossEntropyLoss()
        opt = torch.optim.SGD(
            params=[
                {"params": [module_partial_fc.kernel], "weight_decay": cfg.weight_decay},
                {"params": backbone.parameters()},
            ],
            lr=cfg.lr,
            momentum=cfg.momentum,
        )
    elif cfg.optimizer == "sgd":
        module_partial_fc = PartialFC_V2(
            margin_loss, cfg.embedding_size, cfg.num_classes,
            cfg.sample_rate, False)
        module_partial_fc.train().cuda()
        # TODO the params of partial fc must be last in the params list
        opt = torch.optim.SGD(
            params=[{"params": backbone.parameters()}, {"params": module_partial_fc.parameters()}],
            lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay)

    elif cfg.optimizer == "adamw":
        module_partial_fc = PartialFC_V2(
            margin_loss, cfg.embedding_size, cfg.num_classes,
            cfg.sample_rate, False)
        module_partial_fc.train().cuda()
        opt = torch.optim.AdamW(
            params=[{"params": backbone.parameters()}, {"params": module_partial_fc.parameters()}],
            lr=cfg.lr, weight_decay=cfg.weight_decay)
    else:
        raise

    cfg.total_batch_size = cfg.batch_size * world_size
    cfg.warmup_step = cfg.num_image // cfg.total_batch_size * cfg.warmup_epoch
    cfg.total_step = cfg.num_image // cfg.total_batch_size * cfg.num_epoch
    steps_per_epoch = cfg.num_image // cfg.total_batch_size
    prepbn_decay_epochs = getattr(cfg, "prepbn_decay_epochs", None)
    if prepbn_decay_epochs is not None:
        prepbn_decay_steps = int(steps_per_epoch * prepbn_decay_epochs)
    else:
        prepbn_decay_steps = int(getattr(cfg, "prepbn_decay_steps", cfg.total_step))
    if (getattr(cfg, "prepbn_require_full_transition", False)
            and prepbn_decay_steps > cfg.total_step):
        raise ValueError(
            "RepBatchNorm transition does not finish before training ends: "
            f"decay_steps={prepbn_decay_steps}, total_steps={cfg.total_step}")

    lr_scheduler_name = getattr(cfg, "lr_scheduler", "polynomial")
    lr_scheduler_step_per_epoch = cryptoface_patch_training and lr_scheduler_name == "multistep"
    if lr_scheduler_step_per_epoch:
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            opt,
            milestones=list(getattr(cfg, "lr_milestones", [12, 20, 24])),
            gamma=float(getattr(cfg, "lr_gamma", 0.1)),
        )
    else:
        lr_scheduler = PolynomialLRWarmup(
            optimizer=opt,
            warmup_iters=cfg.warmup_step,
            total_iters=cfg.total_step)

    start_epoch = 0
    global_step = 0
    if cfg.resume:
        dict_checkpoint = torch.load(os.path.join(cfg.output, f"checkpoint_gpu_{rank}.pt"))
        start_epoch = dict_checkpoint["epoch"]
        global_step = dict_checkpoint["global_step"]
        backbone.module.load_state_dict(dict_checkpoint["state_dict_backbone"])
        module_partial_fc.load_state_dict(dict_checkpoint["state_dict_softmax_fc"])
        opt.load_state_dict(dict_checkpoint["state_optimizer"])
        lr_scheduler.load_state_dict(dict_checkpoint["state_lr_scheduler"])
        del dict_checkpoint

    for key, value in cfg.items():
        num_space = 25 - len(key)
        logging.info(": " + key + " " * num_space + str(value))

    callback_verification = CallBackVerification(
        val_targets=cfg.val_targets, rec_prefix=cfg.rec, 
        summary_writer=summary_writer, wandb_logger=wandb_logger,
        fail_on_nonfinite=getattr(cfg, "fail_on_nonfinite_val", False),
        max_embedding_abs=getattr(cfg, "max_validation_embedding_abs", None),
        batch_size=getattr(cfg, "validation_batch_size", 10),
    )
    callback_logging = CallBackLogging(
        frequent=cfg.frequent,
        total_step=cfg.total_step,
        batch_size=cfg.batch_size,
        start_step = global_step,
        writer=summary_writer
    )

    loss_am = AverageMeter()
    amp = torch.cuda.amp.grad_scaler.GradScaler(
        init_scale=float(getattr(cfg, "amp_init_scale", 65536.0)),
        growth_interval=int(getattr(cfg, "amp_growth_interval", 100)),
    )
    grad_clip = float(getattr(cfg, "gradient_clip", 5.0))
    clipped_params = [
        p for group in opt.param_groups for p in group["params"] if p.requires_grad
    ]

    herpn_stage_epochs = tuple(getattr(cfg, "herpn_stage_epochs", ()))
    herpn_conversion_groups = tuple(
        tuple(group) for group in getattr(cfg, "herpn_conversion_groups", ()))
    herpn_group_epochs = tuple(getattr(cfg, "herpn_group_epochs", ()))
    herpn_transition_epochs = float(getattr(cfg, "herpn_transition_epochs", 1.0))
    herpn_range_loss_weight = float(getattr(cfg, "herpn_range_loss_weight", 0.0))
    herpn_distill_loss_weight = float(getattr(cfg, "herpn_distill_loss_weight", 0.0))
    herpn_bn_recalibration_batches = int(
        getattr(cfg, "herpn_bn_recalibration_batches", 0))
    herpn_enabled = hasattr(backbone.module, "set_herpn_progress")
    herpn_group_schedule = bool(herpn_enabled and herpn_conversion_groups)
    if herpn_group_schedule:
        validate_herpn_conversion_groups(
            backbone.module, herpn_conversion_groups)
        final_blends = herpn_group_blends_at_epoch(
            cfg.num_epoch, herpn_conversion_groups, herpn_group_epochs,
            herpn_transition_epochs)
        if (getattr(cfg, "herpn_require_full_conversion", True)
                and min(final_blends.values(), default=0.0) < 1.0):
            raise ValueError(
                "HerPN group schedule does not finish before training ends")
    elif herpn_enabled and herpn_stage_epochs:
        final_progress = herpn_progress_at_epoch(
            cfg.num_epoch, herpn_stage_epochs, herpn_transition_epochs)
        if getattr(cfg, "herpn_require_full_conversion", True) and final_progress < 5.0:
            raise ValueError(
                "HerPN schedule does not finish all five stages before training ends: "
                f"final_progress={final_progress:.3f}"
            )
    completed_herpn_groups = sum(
        float(start_epoch) >= float(start) + herpn_transition_epochs
        for start in herpn_group_epochs
    ) if herpn_group_schedule else 0
    completed_herpn_stages = int(math.floor(float(
        backbone.module.herpn_progress.item()) + 1e-6)
    ) if herpn_enabled and not herpn_group_schedule else 0
    max_steps_per_epoch = int(getattr(cfg, "max_steps_per_epoch", 0))
    scheduled_steps_per_epoch = (
        max_steps_per_epoch if max_steps_per_epoch > 0 else steps_per_epoch)
    simple_gate_stats_interval = int(getattr(
        cfg, "simple_gate_stats_interval", 0))
    simple_gate_enabled = hasattr(
        backbone.module, "set_simple_gate_instrumentation")
    simple_gate_progressive = hasattr(
        backbone.module, "set_simple_gate_blends")
    simple_gate_group_epochs = tuple(getattr(
        cfg, "simple_gate_group_epochs", ()))
    simple_gate_transition_epochs = float(getattr(
        cfg, "simple_gate_transition_epochs", 1.0))
    simple_gate_distill_loss_weight = float(getattr(
        cfg, "simple_gate_distill_loss_weight", 0.0))
    simple_gate_range_loss_weight = float(getattr(
        cfg, "simple_gate_range_loss_weight", 0.0))
    simple_gate_schedule = bool(
        simple_gate_progressive and simple_gate_group_epochs)
    if simple_gate_schedule:
        if (simple_gate_distill_loss_weight <= 0
                and simple_gate_range_loss_weight <= 0):
            raise ValueError(
                "Progressive SimpleGate conversion needs an auxiliary loss "
                "so the multiplier half remains in the DDP graph before its "
                "blend becomes nonzero")
        gate_groups = backbone.module.simple_gate_group_names()
        if len(simple_gate_group_epochs) != len(gate_groups):
            raise ValueError(
                "simple_gate_group_epochs must contain one start for each "
                f"SimpleGate group ({len(gate_groups)}), got "
                f"{len(simple_gate_group_epochs)}")
        first_gate_step = int(simple_gate_group_epochs[0] * steps_per_epoch)
        if first_gate_step < prepbn_decay_steps:
            raise ValueError(
                "SimpleGate conversion overlaps RepBatchNorm transition: "
                f"first_gate_step={first_gate_step}, "
                f"prepbn_decay_steps={prepbn_decay_steps}")
        final_gate_blends = simple_gate_blends_at_epoch(
            cfg.num_epoch, simple_gate_group_epochs,
            simple_gate_transition_epochs)
        if (getattr(cfg, "simple_gate_require_full_conversion", True)
                and min(final_gate_blends, default=0.0) < 1.0):
            raise ValueError(
                "SimpleGate schedule does not finish before training ends")
        set_simple_gate_blends(
            backbone.module,
            simple_gate_blends_at_epoch(
                start_epoch, simple_gate_group_epochs,
                simple_gate_transition_epochs),
        )
    if simple_gate_progressive:
        backbone.module.set_simple_gate_auxiliary_losses(
            simple_gate_distill_loss_weight > 0
            or simple_gate_range_loss_weight > 0)
    completed_simple_gate_groups = sum(
        float(start_epoch) >= float(start) + simple_gate_transition_epochs
        for start in simple_gate_group_epochs
    ) if simple_gate_schedule else 0
    repbn_gate_recalibrated = False
    simple_gate_repbn_recalibration_batches = int(getattr(
        cfg, "simple_gate_repbn_recalibration_batches", 0))
    simple_gate_verify_after_repbn = bool(getattr(
        cfg, "simple_gate_verify_after_repbn", False))
    last_simple_gate_snapshot = {}
    last_simple_gate_snapshot_step = None
    validate_after_prepbn_transition = bool(getattr(
        cfg, "validate_after_prepbn_transition", False))

    for epoch in range(start_epoch, cfg.num_epoch):

        if isinstance(train_loader, DataLoader):
            train_loader.sampler.set_epoch(epoch)
        if (simple_gate_schedule and not repbn_gate_recalibrated
                and prepbn_transition_complete(global_step, prepbn_decay_steps)):
            # Freeze the normalization graph at pure RepBN and evaluate it
            # before the first multiplication gate enters the main path.
            set_prepbn_progress(
                backbone.module, prepbn_decay_steps, prepbn_decay_steps)
            set_simple_gate_blends(
                backbone.module,
                simple_gate_blends_at_epoch(
                    epoch, simple_gate_group_epochs,
                    simple_gate_transition_epochs),
            )
            calibrated = recalibrate_batchnorm_batches(
                backbone, train_loader,
                simple_gate_repbn_recalibration_batches,
                global_step, "post-RepBatchNorm")
            if cfg.dali:
                train_loader.reset()
            if rank == 0:
                logging.info(
                    "RepBatchNorm transition complete; refreshed BN with %d "
                    "batches before SimpleGate conversion", calibrated)
            if simple_gate_verify_after_repbn:
                callback_verification(global_step, backbone.module)
            repbn_gate_recalibrated = True
        if simple_gate_schedule:
            epoch_gate_blends = simple_gate_blends_at_epoch(
                epoch, simple_gate_group_epochs,
                simple_gate_transition_epochs)
            set_simple_gate_blends(backbone.module, epoch_gate_blends)
            newly_completed_gates = sum(
                float(epoch) >= float(start) + simple_gate_transition_epochs
                for start in simple_gate_group_epochs)
            if newly_completed_gates > completed_simple_gate_groups:
                if rank == 0:
                    logging.info(
                        "SimpleGate group %d/%d completed; blends=%s",
                        newly_completed_gates, len(simple_gate_group_epochs),
                        ",".join(f"{value:.3f}" for value in epoch_gate_blends))
                completed_simple_gate_groups = newly_completed_gates
        if herpn_group_schedule:
            epoch_blends = herpn_group_blends_at_epoch(
                epoch, herpn_conversion_groups, herpn_group_epochs,
                herpn_transition_epochs)
            backbone.module.set_herpn_blends(epoch_blends)
            newly_completed = sum(
                float(epoch) >= float(start) + herpn_transition_epochs
                for start in herpn_group_epochs)
            if newly_completed > completed_herpn_groups:
                completed_names = [
                    name
                    for group in herpn_conversion_groups[
                        completed_herpn_groups:newly_completed]
                    for name in group
                ]
                if rank == 0:
                    logging.info(
                        "HerPN group %d/%d completed (%s); recalibrating "
                        "BatchNorm with %d batches",
                        newly_completed, len(herpn_conversion_groups),
                        ", ".join(completed_names),
                        herpn_bn_recalibration_batches)
                recalibrate_herpn_batchnorm(
                    backbone, train_loader, herpn_bn_recalibration_batches,
                    global_step)
                if cfg.dali:
                    train_loader.reset()
                completed_herpn_groups = newly_completed
        elif herpn_enabled and herpn_stage_epochs:
            epoch_herpn_progress = herpn_progress_at_epoch(
                epoch, herpn_stage_epochs, herpn_transition_epochs)
            backbone.module.set_herpn_progress(epoch_herpn_progress)
            newly_completed = int(math.floor(epoch_herpn_progress + 1e-6))
            if newly_completed > completed_herpn_stages:
                if rank == 0:
                    logging.info(
                        "HerPN stage %d/5 completed; recalibrating BatchNorm with %d batches",
                        newly_completed, herpn_bn_recalibration_batches)
                recalibrate_herpn_batchnorm(
                    backbone, train_loader, herpn_bn_recalibration_batches, global_step)
                if cfg.dali:
                    train_loader.reset()
                completed_herpn_stages = newly_completed
        for step_in_epoch, (img, local_labels) in enumerate(train_loader):
            if max_steps_per_epoch > 0 and step_in_epoch >= max_steps_per_epoch:
                break
            global_step += 1
            set_prepbn_progress(backbone.module, global_step, prepbn_decay_steps)
            capture_simple_gate_stats = (
                simple_gate_enabled
                and simple_gate_stats_interval > 0
                and global_step % simple_gate_stats_interval == 0
            )
            set_simple_gate_instrumentation(
                backbone.module,
                capture_simple_gate_stats,
                gradient_scale=float(amp.get_scale()) if cfg.fp16 else 1.0,
            )
            if herpn_group_schedule:
                fractional_epoch = epoch + step_in_epoch / max(
                    scheduled_steps_per_epoch, 1)
                backbone.module.set_herpn_blends(herpn_group_blends_at_epoch(
                    fractional_epoch, herpn_conversion_groups,
                    herpn_group_epochs, herpn_transition_epochs))
            elif herpn_enabled and herpn_stage_epochs:
                fractional_epoch = epoch + step_in_epoch / max(
                    scheduled_steps_per_epoch, 1)
                backbone.module.set_herpn_progress(herpn_progress_at_epoch(
                    fractional_epoch, herpn_stage_epochs, herpn_transition_epochs))
            if simple_gate_schedule:
                fractional_epoch = epoch + step_in_epoch / max(
                    scheduled_steps_per_epoch, 1)
                set_simple_gate_blends(
                    backbone.module,
                    simple_gate_blends_at_epoch(
                        fractional_epoch, simple_gate_group_epochs,
                        simple_gate_transition_epochs),
                )
            backbone_output = backbone(img)
            if cryptoface_patch_training:
                local_embeddings, patch_pred, patch_target = backbone_output
            else:
                local_embeddings = backbone_output
            if not torch.isfinite(local_embeddings).all():
                gate_context = ""
                if last_simple_gate_snapshot:
                    worst_name, worst_stats = max(
                        last_simple_gate_snapshot.items(),
                        key=lambda item: item[1]["product_absmax"])
                    gate_context = (
                        f"; last_gate_profile_step="
                        f"{last_simple_gate_snapshot_step}, "
                        f"worst_gate={worst_name}, "
                        f"product_absmax={worst_stats['product_absmax']:.6g}, "
                        f"product_p999={worst_stats['product_p999']:.6g}, "
                        f"blend={worst_stats.get('blend', float('nan')):.3f}")
                raise FloatingPointError(
                    f"Non-finite embeddings at global_step={global_step}"
                    f"{gate_context}")
            if cryptoface_patch_training:
                local_labels = local_labels.squeeze().long()
                logits = module_partial_fc(local_embeddings, local_labels)
                loss: torch.Tensor = criterion(logits, local_labels)
                loss_jigsaw = F.cross_entropy(patch_pred, patch_target)
                loss = loss + float(getattr(cfg, "patch_cnn_jigsaw_weight", 0.005)) * loss_jigsaw
            else:
                loss: torch.Tensor = module_partial_fc(local_embeddings, local_labels)
            range_penalty = local_embeddings.new_zeros(())
            distillation_loss = local_embeddings.new_zeros(())
            simple_gate_range_penalty = local_embeddings.new_zeros(())
            simple_gate_distillation_loss = local_embeddings.new_zeros(())
            if herpn_enabled and herpn_range_loss_weight > 0:
                range_penalty = backbone.module.herpn_range_penalty()
                if not torch.isfinite(range_penalty):
                    raise FloatingPointError(
                        f"Non-finite HerPN range penalty at global_step={global_step}"
                    )
                loss = loss + herpn_range_loss_weight * range_penalty
            if herpn_enabled and herpn_distill_loss_weight > 0:
                distillation_loss = backbone.module.herpn_distillation_loss()
                if not torch.isfinite(distillation_loss):
                    raise FloatingPointError(
                        f"Non-finite HerPN distillation loss at global_step={global_step}"
                    )
                loss = loss + herpn_distill_loss_weight * distillation_loss
            if simple_gate_progressive and simple_gate_range_loss_weight > 0:
                simple_gate_range_penalty = (
                    backbone.module.simple_gate_range_penalty())
                if not torch.isfinite(simple_gate_range_penalty):
                    raise FloatingPointError(
                        "Non-finite SimpleGate range penalty at "
                        f"global_step={global_step}")
                loss = loss + (
                    simple_gate_range_loss_weight * simple_gate_range_penalty)
            if simple_gate_progressive and simple_gate_distill_loss_weight > 0:
                simple_gate_distillation_loss = (
                    backbone.module.simple_gate_distillation_loss())
                if not torch.isfinite(simple_gate_distillation_loss):
                    raise FloatingPointError(
                        "Non-finite SimpleGate distillation loss at "
                        f"global_step={global_step}")
                loss = loss + (
                    simple_gate_distill_loss_weight
                    * simple_gate_distillation_loss)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at global_step={global_step}: {loss.item()}")

            if cfg.fp16:
                amp.scale(loss).backward()
                if global_step % cfg.gradient_acc == 0:
                    amp.unscale_(opt)
                    if getattr(cfg, "check_finite_grads", False):
                        check_finite_gradients(backbone, "backbone", global_step)
                        check_finite_gradients(module_partial_fc, "partial_fc", global_step)
                    if getattr(cfg, "gradient_clip_type", "norm") == "value":
                        torch.nn.utils.clip_grad_value_(clipped_params, grad_clip)
                        total_norm = torch.tensor(0.0, device=local_embeddings.device)
                    else:
                        total_norm = torch.nn.utils.clip_grad_norm_(
                            clipped_params, grad_clip, error_if_nonfinite=False
                        )
                    if getattr(cfg, "gradient_clip_type", "norm") == "value" or torch.isfinite(total_norm):
                        amp.step(opt)
                    else:
                        logging.warning(
                            "Skipping optimizer step at global_step=%d due to non-finite grad norm: %s",
                            global_step, total_norm.item()
                        )
                    amp.update()
                    opt.zero_grad()
            else:
                loss.backward()
                if global_step % cfg.gradient_acc == 0:
                    check_finite_gradients(backbone, "backbone", global_step)
                    check_finite_gradients(module_partial_fc, "partial_fc", global_step)
                    if getattr(cfg, "gradient_clip_type", "norm") == "value":
                        torch.nn.utils.clip_grad_value_(clipped_params, grad_clip)
                    else:
                        torch.nn.utils.clip_grad_norm_(clipped_params, grad_clip, error_if_nonfinite=True)
                    opt.step()
                    opt.zero_grad()
            if not lr_scheduler_step_per_epoch:
                lr_scheduler.step()

            with torch.no_grad():
                if capture_simple_gate_stats:
                    gate_stats = collect_simple_gate_stats(backbone.module)
                    last_simple_gate_snapshot = log_simple_gate_stats(
                        gate_stats,
                        global_step,
                        summary_writer=summary_writer,
                        wandb_logger=wandb_logger,
                    )
                    last_simple_gate_snapshot_step = global_step
                    set_simple_gate_instrumentation(backbone.module, False)
                if wandb_logger:
                    wandb_logger.log({
                        'Loss/Step Loss': loss.item(),
                        'Loss/Train Loss': loss_am.avg,
                        'Loss/HerPN Range Penalty': range_penalty.item(),
                        'Loss/HerPN Distillation': distillation_loss.item(),
                        'Loss/SimpleGate Range Penalty': (
                            simple_gate_range_penalty.item()),
                        'Loss/SimpleGate Distillation': (
                            simple_gate_distillation_loss.item()),
                        'Process/SimpleGate Progress': (
                            sum(simple_gate_blends_at_epoch(
                                epoch + step_in_epoch / max(
                                    scheduled_steps_per_epoch, 1),
                                simple_gate_group_epochs,
                                simple_gate_transition_epochs))
                            if simple_gate_schedule else 0.0),
                        'Process/HerPN Progress': (
                            float(backbone.module.herpn_progress.item())
                            if herpn_enabled else 0.0),
                        'Process/Step': global_step,
                        'Process/Epoch': epoch
                    })

                if (summary_writer is not None and herpn_enabled and
                        global_step % cfg.frequent == 0):
                    range_summary = backbone.module.herpn_range_summary()
                    summary_writer.add_scalar(
                        'Loss/HerPN Range Penalty', range_penalty.item(), global_step)
                    summary_writer.add_scalar(
                        'Loss/HerPN Distillation', distillation_loss.item(), global_step)
                    summary_writer.add_scalar(
                        'Process/HerPN Progress',
                        float(backbone.module.herpn_progress.item()), global_step)
                    summary_writer.add_scalar(
                        'HerPN/Input Abs Max',
                        float(range_summary['input_absmax'].item()), global_step)
                    summary_writer.add_scalar(
                        'HerPN/Outside Range Fraction',
                        float(range_summary['outside_fraction'].item()), global_step)
                if (summary_writer is not None and simple_gate_progressive
                        and global_step % cfg.frequent == 0):
                    summary_writer.add_scalar(
                        'Loss/SimpleGate Range Penalty',
                        simple_gate_range_penalty.item(), global_step)
                    summary_writer.add_scalar(
                        'Loss/SimpleGate Distillation',
                        simple_gate_distillation_loss.item(), global_step)
                    if simple_gate_schedule:
                        current_gate_blends = simple_gate_blends_at_epoch(
                            epoch + step_in_epoch / max(
                                scheduled_steps_per_epoch, 1),
                            simple_gate_group_epochs,
                            simple_gate_transition_epochs)
                        for group_index, blend in enumerate(
                                current_gate_blends):
                            summary_writer.add_scalar(
                                f'Process/SimpleGate Group {group_index} Blend',
                                blend, global_step)
                if (summary_writer is not None and prepbn_decay_steps > 0
                        and global_step % cfg.frequent == 0):
                    summary_writer.add_scalar(
                        'RepBatchNorm/Transition Progress',
                        min(global_step / prepbn_decay_steps, 1.0),
                        global_step,
                    )
                    
                loss_am.update(loss.item(), 1)
                callback_logging(global_step, loss_am, epoch, cfg.fp16, lr_scheduler.get_last_lr()[0], amp)

                if global_step % cfg.verbose == 0 and global_step > 0:
                    if rank == 0 and getattr(cfg, "save_validation_snapshots", False):
                        torch.save(
                            backbone.module.state_dict(),
                            os.path.join(cfg.output, "model_validation.pt"),
                        )
                    if (validate_after_prepbn_transition
                            and not prepbn_transition_complete(
                                global_step, prepbn_decay_steps)):
                        if rank == 0:
                            logging.info(
                                "Skipping verification at step %d: RepBatchNorm "
                                "transition is %.2f%% complete",
                                global_step,
                                100.0 * global_step / max(prepbn_decay_steps, 1),
                            )
                    else:
                        callback_verification(global_step, backbone.module)

        if lr_scheduler_step_per_epoch:
            lr_scheduler.step()

        checkpoint_interval = int(getattr(cfg, "checkpoint_interval_epochs", 0))
        if (cfg.save_all_states or
                (checkpoint_interval > 0
                 and (epoch + 1) % checkpoint_interval == 0)):
            checkpoint = {
                "epoch": epoch + 1,
                "global_step": global_step,
                "state_dict_backbone": backbone.module.state_dict(),
                "state_dict_softmax_fc": module_partial_fc.state_dict(),
                "state_optimizer": opt.state_dict(),
                "state_lr_scheduler": lr_scheduler.state_dict()
            }
            atomic_torch_save(
                checkpoint,
                os.path.join(cfg.output, f"checkpoint_gpu_{rank}.pt"))

        if rank == 0:
            path_module = os.path.join(cfg.output, "model.pt")
            backbone_state = backbone.module.state_dict()
            atomic_torch_save(backbone_state, path_module)
            epoch_model_interval = int(getattr(cfg, "epoch_model_interval", 0))
            if (getattr(cfg, "save_epoch_models", False)
                    and epoch_model_interval > 0
                    and (epoch + 1) % epoch_model_interval == 0):
                epoch_model_path = os.path.join(
                    cfg.output, f"model_epoch_{epoch + 1:02d}.pt")
                atomic_torch_save(backbone_state, epoch_model_path)
                logging.info(
                    "Saved inference snapshot for epoch %d to %s",
                    epoch + 1, epoch_model_path)

            if wandb_logger and cfg.save_artifacts:
                artifact_name = f"{run_name}_E{epoch}"
                model = wandb.Artifact(artifact_name, type='model')
                model.add_file(path_module)
                wandb_logger.log_artifact(model)
                
        if cfg.dali:
            train_loader.reset()

    if simple_gate_schedule:
        set_simple_gate_blends(
            backbone.module, (1.0,) * len(simple_gate_group_epochs))

    prepbn_bn_stat_epochs = int(getattr(cfg, "prepbn_bn_stat_epochs", 0))
    if prepbn_bn_stat_epochs > 0:
        if rank == 0:
            logging.info("Refreshing PRepBN BatchNorm statistics for %d epoch(s)", prepbn_bn_stat_epochs)
        set_prepbn_progress(backbone.module, prepbn_decay_steps, prepbn_decay_steps)
        recalibration_batches = recalibrate_prepbn_batchnorm(
            backbone,
            train_loader,
            prepbn_bn_stat_epochs,
            cfg.num_epoch,
            dali=cfg.dali,
        )
        if rank == 0:
            logging.info(
                "Refreshed final RepBatchNorm statistics with %d batches",
                recalibration_batches,
            )

    final_gate_profile = profile_simple_gate_ranges(
        backbone,
        train_loader,
        int(getattr(cfg, "simple_gate_final_profile_batches", 0)),
        dali=cfg.dali,
    )
    if final_gate_profile and rank == 0:
        profile_path = os.path.join(cfg.output, "simple_gate_final_profile.json")
        with open(profile_path, "w", encoding="utf-8") as profile_file:
            json.dump(final_gate_profile, profile_file, indent=2, sort_keys=True)
        profile_layers = {
            name: stats for name, stats in final_gate_profile.items()
            if not name.startswith("_")
        }
        worst_name, worst_stats = max(
            profile_layers.items(),
            key=lambda item: item[1]["product_absmax"],
        )
        logging.info(
            "Final SimpleGate profile saved to %s; worst product range is "
            "%s absmax=%.6g p99.9=%.6g outside=%.6g",
            profile_path,
            worst_name,
            worst_stats["product_absmax"],
            worst_stats["product_p999"],
            worst_stats["product_outside_fraction"],
        )

    if getattr(cfg, "final_verification_after_prepbn", False):
        if rank == 0:
            logging.info(
                "Running final verification with fully converted and "
                "recalibrated RepBatchNorm")
        callback_verification(global_step, backbone.module)

    if rank == 0:
        path_module = os.path.join(cfg.output, "model.pt")
        atomic_torch_save(backbone.module.state_dict(), path_module)
        
        if wandb_logger and cfg.save_artifacts:
            artifact_name = f"{run_name}_Final"
            model = wandb.Artifact(artifact_name, type='model')
            model.add_file(path_module)
            wandb_logger.log_artifact(model)



if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser(
        description="Distributed Arcface Training in Pytorch")
    parser.add_argument("config", type=str, help="py config file")
    main(parser.parse_args())

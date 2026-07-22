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


def set_simple_gate_instrumentation(module: torch.nn.Module, enabled: bool,
                                    gradient_scale: float = 1.0):
    setter = getattr(module, "set_simple_gate_instrumentation", None)
    if setter is not None:
        setter(enabled, gradient_scale=gradient_scale)


def collect_simple_gate_stats(module: torch.nn.Module):
    collector = getattr(module, "simple_gate_range_stats", None)
    return collector() if collector is not None else {}


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
    """Return per-activation blends for an arbitrary block-wise schedule."""
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
    scheduled = {
        name for group in conversion_groups for name in group
    }
    missing = sorted(expected.difference(scheduled))
    unknown = sorted(scheduled.difference(expected))
    count = sum(len(group) for group in conversion_groups)
    if count != len(scheduled):
        raise ValueError("An activation occurs more than once in HerPN conversion groups")
    if missing or unknown:
        raise ValueError(
            f"Invalid HerPN conversion groups; missing={missing}, unknown={unknown}")


def build_backbone_optimizer_groups(backbone, cfg):
    """Give HerPN coefficients a lower LR and PReLU teachers no decay."""
    herpn_params = []
    teacher_params = []
    for module in backbone.module.modules():
        if module.__class__.__name__ != "ProgressiveHerPNActivation":
            continue
        herpn_params.extend(module.herpn.parameters())
        teacher_params.extend(module.prelu.parameters())

    special_ids = {id(param) for param in herpn_params + teacher_params}
    base_params = [
        param for param in backbone.parameters()
        if param.requires_grad and id(param) not in special_ids
    ]
    if not herpn_params:
        return [{"params": base_params}]
    return [
        {"params": base_params},
        {
            "params": herpn_params,
            "lr": cfg.lr * float(getattr(cfg, "herpn_lr_multiplier", 0.1)),
            "weight_decay": float(getattr(
                cfg, "herpn_weight_decay", cfg.weight_decay)),
        },
        {
            "params": teacher_params,
            "weight_decay": 0.0,
            # With no momentum, a fully converted branch's exact zero task
            # gradient keeps its PReLU teacher fixed for persistent distillation.
            "momentum": float(getattr(cfg, "herpn_teacher_momentum", 0.0)),
        },
    ]


def save_training_checkpoint(checkpoint, output, rank, keep_previous=True):
    """Atomically save a resumable checkpoint and retain the previous epoch."""
    path = os.path.join(output, f"checkpoint_gpu_{rank}.pt")
    temporary_path = path + ".tmp"
    previous_path = path + ".previous"
    torch.save(checkpoint, temporary_path)
    if keep_previous and os.path.exists(path):
        os.replace(path, previous_path)
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
            herpn_output_limit=float(getattr(cfg, "herpn_output_limit", 8.0)),
            herpn_bn_eps=float(getattr(cfg, "herpn_bn_eps", 1e-4)),
            herpn_quadratic_bn_eps=float(getattr(
                cfg, "herpn_quadratic_bn_eps", 1e-3)),
            herpn_distillation_floor=float(getattr(
                cfg, "herpn_distillation_floor", 0.1)),
            herpn_progress=float(getattr(
                cfg, "herpn_initial_progress", default_herpn_progress)),
        )
    if cfg.network.startswith("poolformer_no_ln_x2_act"):
        model_kwargs.update(
            gate_range_limit=float(getattr(cfg, "simple_gate_range_limit", 6.0)),
            gate_stats_sample_size=int(getattr(
                cfg, "simple_gate_stats_sample_size", 16384)),
            gate_compute_fp32=bool(getattr(
                cfg, "simple_gate_compute_fp32", True)),
            gate_fail_on_nonfinite=bool(getattr(
                cfg, "simple_gate_fail_on_nonfinite", True)),
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
        backbone_optimizer_groups = build_backbone_optimizer_groups(backbone, cfg)
        opt = torch.optim.SGD(
            params=backbone_optimizer_groups + [
                {"params": module_partial_fc.parameters()}],
            lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay)

    elif cfg.optimizer == "adamw":
        module_partial_fc = PartialFC_V2(
            margin_loss, cfg.embedding_size, cfg.num_classes,
            cfg.sample_rate, False)
        module_partial_fc.train().cuda()
        backbone_optimizer_groups = build_backbone_optimizer_groups(backbone, cfg)
        opt = torch.optim.AdamW(
            params=backbone_optimizer_groups + [
                {"params": module_partial_fc.parameters()}],
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
        resume_path = getattr(
            cfg, "resume_checkpoint",
            os.path.join(cfg.output, f"checkpoint_gpu_{rank}.pt"))
        resume_path = str(resume_path).format(rank=rank)
        dict_checkpoint = torch.load(resume_path, map_location="cpu")
        start_epoch = dict_checkpoint["epoch"]
        global_step = dict_checkpoint["global_step"]
        backbone.module.load_state_dict(dict_checkpoint["state_dict_backbone"])
        module_partial_fc.load_state_dict(dict_checkpoint["state_dict_softmax_fc"])
        opt.load_state_dict(dict_checkpoint["state_optimizer"])
        lr_scheduler.load_state_dict(dict_checkpoint["state_lr_scheduler"])
        logging.info("Resumed complete training state from %s", resume_path)
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
    herpn_output_loss_weight = float(getattr(cfg, "herpn_output_loss_weight", 0.0))
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
    validate_after_prepbn_transition = bool(getattr(
        cfg, "validate_after_prepbn_transition", False))

    for epoch in range(start_epoch, cfg.num_epoch):

        if isinstance(train_loader, DataLoader):
            train_loader.sampler.set_epoch(epoch)
        if herpn_group_schedule:
            epoch_herpn_blends = herpn_group_blends_at_epoch(
                epoch, herpn_conversion_groups, herpn_group_epochs,
                herpn_transition_epochs)
            backbone.module.set_herpn_blends(epoch_herpn_blends)
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
                        "HerPN groups %d/%d completed (%s); recalibrating "
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
            backbone_output = backbone(img)
            if cryptoface_patch_training:
                local_embeddings, patch_pred, patch_target = backbone_output
            else:
                local_embeddings = backbone_output
            if not torch.isfinite(local_embeddings).all():
                raise FloatingPointError(f"Non-finite embeddings at global_step={global_step}")
            if cryptoface_patch_training:
                local_labels = local_labels.squeeze().long()
                logits = module_partial_fc(local_embeddings, local_labels)
                loss: torch.Tensor = criterion(logits, local_labels)
                loss_jigsaw = F.cross_entropy(patch_pred, patch_target)
                loss = loss + float(getattr(cfg, "patch_cnn_jigsaw_weight", 0.005)) * loss_jigsaw
            else:
                loss: torch.Tensor = module_partial_fc(local_embeddings, local_labels)
            range_penalty = local_embeddings.new_zeros(())
            output_penalty = local_embeddings.new_zeros(())
            distillation_loss = local_embeddings.new_zeros(())
            if herpn_enabled and herpn_range_loss_weight > 0:
                range_penalty = backbone.module.herpn_range_penalty()
                if not torch.isfinite(range_penalty):
                    raise FloatingPointError(
                        f"Non-finite HerPN range penalty at global_step={global_step}"
                    )
                loss = loss + herpn_range_loss_weight * range_penalty
            if herpn_enabled and herpn_output_loss_weight > 0:
                output_penalty = backbone.module.herpn_output_penalty()
                if not torch.isfinite(output_penalty):
                    raise FloatingPointError(
                        f"Non-finite HerPN output penalty at global_step={global_step}"
                    )
                loss = loss + herpn_output_loss_weight * output_penalty
            if herpn_enabled and herpn_distill_loss_weight > 0:
                distillation_loss = backbone.module.herpn_distillation_loss()
                if not torch.isfinite(distillation_loss):
                    raise FloatingPointError(
                        f"Non-finite HerPN distillation loss at global_step={global_step}"
                    )
                loss = loss + herpn_distill_loss_weight * distillation_loss
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
                    log_simple_gate_stats(
                        gate_stats,
                        global_step,
                        summary_writer=summary_writer,
                        wandb_logger=wandb_logger,
                    )
                    set_simple_gate_instrumentation(backbone.module, False)
                if wandb_logger:
                    wandb_logger.log({
                        'Loss/Step Loss': loss.item(),
                        'Loss/Train Loss': loss_am.avg,
                        'Loss/HerPN Range Penalty': range_penalty.item(),
                        'Loss/HerPN Output Penalty': output_penalty.item(),
                        'Loss/HerPN Distillation': distillation_loss.item(),
                        'Process/HerPN Progress': (
                            float(backbone.module.herpn_progress.item())
                            if herpn_enabled else 0.0),
                        'Process/Step': global_step,
                        'Process/Epoch': epoch
                    })

                if (summary_writer is not None and herpn_enabled and
                        global_step % cfg.frequent == 0):
                    layer_range_stats = backbone.module.herpn_range_stats()
                    range_summary = backbone.module.herpn_range_summary()
                    summary_writer.add_scalar(
                        'Loss/HerPN Range Penalty', range_penalty.item(), global_step)
                    summary_writer.add_scalar(
                        'Loss/HerPN Output Penalty', output_penalty.item(), global_step)
                    summary_writer.add_scalar(
                        'Loss/HerPN Distillation', distillation_loss.item(), global_step)
                    summary_writer.add_scalar(
                        'Process/HerPN Progress',
                        float(backbone.module.herpn_progress.item()), global_step)
                    summary_writer.add_scalar(
                        'HerPN/Input Abs Max',
                        float(range_summary['input_absmax'].item()), global_step)
                    summary_writer.add_scalar(
                        'HerPN/Output Abs Max',
                        float(range_summary['output_absmax'].item()), global_step)
                    summary_writer.add_scalar(
                        'HerPN/Outside Range Fraction',
                        float(range_summary['outside_fraction'].item()), global_step)
                    summary_writer.add_scalar(
                        'HerPN/Output Outside Range Fraction',
                        float(range_summary['output_outside_fraction'].item()),
                        global_step)
                    summary_writer.add_scalar(
                        'HerPN/Weight Abs Max',
                        float(range_summary['herpn_weight_absmax'].item()), global_step)
                    summary_writer.add_scalar(
                        'HerPN/BN2 Running Var Min',
                        float(range_summary['bn2_running_var_min'].item()), global_step)
                    summary_writer.add_scalar(
                        'HerPN/Quadratic Coefficient Abs Max',
                        float(range_summary['coefficient2_absmax'].item()), global_step)
                    for layer_name, layer_stats in layer_range_stats.items():
                        layer_tag = layer_name.replace('.', '/')
                        for metric in (
                                'output_absmax', 'herpn_weight_absmax',
                                'bn2_running_var_min', 'coefficient2_absmax',
                                'blend'):
                            value = layer_stats[metric]
                            if value is not None:
                                summary_writer.add_scalar(
                                    f'HerPN/Layers/{layer_tag}/{metric}',
                                    float(value.item()), global_step)
                    worst_output_name, worst_output_stats = max(
                        layer_range_stats.items(),
                        key=lambda item: float(
                            item[1]['output_absmax'].item())
                        if item[1]['output_absmax'] is not None else -1.0)
                    worst_coefficient_name, worst_coefficient_stats = max(
                        layer_range_stats.items(),
                        key=lambda item: float(
                            item[1]['coefficient2_absmax'].item()))
                    logging.info(
                        "HerPN stability step=%d output=%s:%.6g "
                        "coefficient2=%s:%.6g bn2_var_min=%.6g",
                        global_step,
                        worst_output_name,
                        float(worst_output_stats['output_absmax'].item()),
                        worst_coefficient_name,
                        float(worst_coefficient_stats[
                            'coefficient2_absmax'].item()),
                        float(range_summary['bn2_running_var_min'].item()))
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
        should_save_checkpoint = (
            bool(cfg.save_all_states)
            or (checkpoint_interval > 0
                and (epoch + 1) % checkpoint_interval == 0)
        )
        if should_save_checkpoint:
            checkpoint = {
                "epoch": epoch + 1,
                "global_step": global_step,
                "state_dict_backbone": backbone.module.state_dict(),
                "state_dict_softmax_fc": module_partial_fc.state_dict(),
                "state_optimizer": opt.state_dict(),
                "state_lr_scheduler": lr_scheduler.state_dict()
            }
            save_training_checkpoint(
                checkpoint, cfg.output, rank,
                keep_previous=bool(getattr(
                    cfg, "checkpoint_keep_previous", True)))
            logging.info(
                "Saved resumable checkpoint for epoch %d on rank %d",
                epoch + 1, rank)

        if rank == 0:
            path_module = os.path.join(cfg.output, "model.pt")
            torch.save(backbone.module.state_dict(), path_module)

            if wandb_logger and cfg.save_artifacts:
                artifact_name = f"{run_name}_E{epoch}"
                model = wandb.Artifact(artifact_name, type='model')
                model.add_file(path_module)
                wandb_logger.log_artifact(model)
                
        if cfg.dali:
            train_loader.reset()

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
        torch.save(backbone.module.state_dict(), path_module)
        
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

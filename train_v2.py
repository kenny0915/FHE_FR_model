import argparse
import logging
import math
import os
from datetime import datetime

import numpy as np
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
            0.0 if getattr(cfg, "herpn_stage_epochs", ()) else 5.0)
        model_kwargs.update(
            herpn_range_limit=float(getattr(cfg, "herpn_range_limit", 6.0)),
            herpn_bn_eps=float(getattr(cfg, "herpn_bn_eps", 1e-4)),
            herpn_progress=float(getattr(
                cfg, "herpn_initial_progress", default_herpn_progress)),
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
    herpn_transition_epochs = float(getattr(cfg, "herpn_transition_epochs", 1.0))
    herpn_range_loss_weight = float(getattr(cfg, "herpn_range_loss_weight", 0.0))
    herpn_distill_loss_weight = float(getattr(cfg, "herpn_distill_loss_weight", 0.0))
    herpn_bn_recalibration_batches = int(
        getattr(cfg, "herpn_bn_recalibration_batches", 0))
    herpn_enabled = hasattr(backbone.module, "set_herpn_progress")
    if herpn_enabled and herpn_stage_epochs:
        final_progress = herpn_progress_at_epoch(
            cfg.num_epoch, herpn_stage_epochs, herpn_transition_epochs)
        if getattr(cfg, "herpn_require_full_conversion", True) and final_progress < 5.0:
            raise ValueError(
                "HerPN schedule does not finish all five stages before training ends: "
                f"final_progress={final_progress:.3f}"
            )
    completed_herpn_stages = int(math.floor(float(
        backbone.module.herpn_progress.item()) + 1e-6)) if herpn_enabled else 0
    max_steps_per_epoch = int(getattr(cfg, "max_steps_per_epoch", 0))
    scheduled_steps_per_epoch = (
        max_steps_per_epoch if max_steps_per_epoch > 0 else steps_per_epoch)

    for epoch in range(start_epoch, cfg.num_epoch):

        if isinstance(train_loader, DataLoader):
            train_loader.sampler.set_epoch(epoch)
        if herpn_enabled and herpn_stage_epochs:
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
            if herpn_enabled and herpn_stage_epochs:
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
            distillation_loss = local_embeddings.new_zeros(())
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
                if wandb_logger:
                    wandb_logger.log({
                        'Loss/Step Loss': loss.item(),
                        'Loss/Train Loss': loss_am.avg,
                        'Loss/HerPN Range Penalty': range_penalty.item(),
                        'Loss/HerPN Distillation': distillation_loss.item(),
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
                    
                loss_am.update(loss.item(), 1)
                callback_logging(global_step, loss_am, epoch, cfg.fp16, lr_scheduler.get_last_lr()[0], amp)

                if global_step % cfg.verbose == 0 and global_step > 0:
                    if rank == 0 and getattr(cfg, "save_validation_snapshots", False):
                        torch.save(
                            backbone.module.state_dict(),
                            os.path.join(cfg.output, "model_validation.pt"),
                        )
                    callback_verification(global_step, backbone)

        if lr_scheduler_step_per_epoch:
            lr_scheduler.step()

        if cfg.save_all_states:
            checkpoint = {
                "epoch": epoch + 1,
                "global_step": global_step,
                "state_dict_backbone": backbone.module.state_dict(),
                "state_dict_softmax_fc": module_partial_fc.state_dict(),
                "state_optimizer": opt.state_dict(),
                "state_lr_scheduler": lr_scheduler.state_dict()
            }
            torch.save(checkpoint, os.path.join(cfg.output, f"checkpoint_gpu_{rank}.pt"))

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
        backbone.train()
        set_prepbn_progress(backbone.module, prepbn_decay_steps, prepbn_decay_steps)
        with torch.no_grad():
            for stat_epoch in range(prepbn_bn_stat_epochs):
                if isinstance(train_loader, DataLoader):
                    train_loader.sampler.set_epoch(cfg.num_epoch + stat_epoch)
                for img, _ in train_loader:
                    backbone(img)
                if cfg.dali:
                    train_loader.reset()

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

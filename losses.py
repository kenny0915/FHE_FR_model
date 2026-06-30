import torch
import math


class CombinedMarginLoss(torch.nn.Module):
    def __init__(self, 
                 s, 
                 m1,
                 m2,
                 m3,
                 interclass_filtering_threshold=0):
        super().__init__()
        self.s = s
        self.m1 = m1
        self.m2 = m2
        self.m3 = m3
        self.interclass_filtering_threshold = interclass_filtering_threshold
        
        # For ArcFace
        self.cos_m = math.cos(self.m2)
        self.sin_m = math.sin(self.m2)
        self.theta = math.cos(math.pi - self.m2)
        self.sinmm = math.sin(math.pi - self.m2) * self.m2
        self.easy_margin = False


    def forward(self, logits, labels, norms=None):
        index_positive = torch.where(labels != -1)[0]

        if self.interclass_filtering_threshold > 0:
            with torch.no_grad():
                dirty = logits > self.interclass_filtering_threshold
                dirty = dirty.float()
                mask = torch.ones([index_positive.size(0), logits.size(1)], device=logits.device)
                mask.scatter_(1, labels[index_positive], 0)
                dirty[index_positive] *= mask
                tensor_mul = 1 - dirty    
            logits = tensor_mul * logits

        target_logit = logits[index_positive, labels[index_positive].view(-1)]

        if self.m1 == 1.0 and self.m3 == 0.0:
            with torch.no_grad():
                target_logit.arccos_()
                logits.arccos_()
                final_target_logit = target_logit + self.m2
                logits[index_positive, labels[index_positive].view(-1)] = final_target_logit
                logits.cos_()
            logits = logits * self.s        

        elif self.m3 > 0:
            final_target_logit = target_logit - self.m3
            logits[index_positive, labels[index_positive].view(-1)] = final_target_logit
            logits = logits * self.s
        else:
            raise

        return logits

class ArcFace(torch.nn.Module):
    """ ArcFace (https://arxiv.org/pdf/1801.07698v1.pdf):
    """
    def __init__(self, s=64.0, margin=0.5):
        super(ArcFace, self).__init__()
        self.s = s
        self.margin = margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.theta = math.cos(math.pi - margin)
        self.sinmm = math.sin(math.pi - margin) * margin
        self.easy_margin = False


    def forward(self, logits: torch.Tensor, labels: torch.Tensor, norms=None):
        index = torch.where(labels != -1)[0]
        target_logit = logits[index, labels[index].view(-1)]

        with torch.no_grad():
            target_logit.arccos_()
            logits.arccos_()
            final_target_logit = target_logit + self.margin
            logits[index, labels[index].view(-1)] = final_target_logit
            logits.cos_()
        logits = logits * self.s   
        return logits


class CosFace(torch.nn.Module):
    def __init__(self, s=64.0, m=0.40):
        super(CosFace, self).__init__()
        self.s = s
        self.m = m

    def forward(self, logits: torch.Tensor, labels: torch.Tensor, norms=None):
        index = torch.where(labels != -1)[0]
        target_logit = logits[index, labels[index].view(-1)]
        final_target_logit = target_logit - self.m
        logits[index, labels[index].view(-1)] = final_target_logit
        logits = logits * self.s
        return logits

class AdaFace(torch.nn.Module):
    """AdaFace margin from https://github.com/mk-minchul/AdaFace.

    This module is written to plug into PartialFC_V2: it receives cosine logits
    from normalized embeddings/weights, labels in the local class partition, and
    raw embedding norms gathered across all ranks.
    """

    def __init__(self, s=64.0, margin=0.4, h=0.333, t_alpha=1.0, eps=1e-3):
        super(AdaFace, self).__init__()
        self.s = s
        self.margin = margin
        self.h = h
        self.t_alpha = t_alpha
        self.eps = eps
        self.register_buffer("batch_mean", torch.ones(1) * 20)
        self.register_buffer("batch_std", torch.ones(1) * 100)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor, norms: torch.Tensor):
        if norms is None:
            raise ValueError("AdaFace requires raw embedding norms from PartialFC_V2.")

        index = torch.where(labels != -1)[0]
        if index.numel() == 0:
            return logits * self.s

        safe_norms = torch.clip(norms, min=0.001, max=100).detach()
        with torch.no_grad():
            batch_mean = safe_norms.mean()
            batch_std = safe_norms.std()
            self.batch_mean.mul_(1 - self.t_alpha).add_(batch_mean * self.t_alpha)
            self.batch_std.mul_(1 - self.t_alpha).add_(batch_std * self.t_alpha)

        margin_scaler = (safe_norms - self.batch_mean) / (self.batch_std + self.eps)
        margin_scaler = torch.clip(margin_scaler * self.h, -1, 1)

        target_logit = logits[index, labels[index].view(-1)]
        target_scaler = margin_scaler[index].view(-1)

        # Angular component: low-quality samples receive a larger angular margin.
        g_angular = -self.margin * target_scaler
        theta = target_logit.clamp(-1 + self.eps, 1 - self.eps).acos()
        theta = torch.clip(theta + g_angular, min=self.eps, max=math.pi - self.eps)
        final_target_logit = theta.cos()

        # Additive component from AdaFace.
        g_additive = self.margin + (self.margin * target_scaler)
        final_target_logit = final_target_logit - g_additive

        logits[index, labels[index].view(-1)] = final_target_logit
        logits = logits * self.s
        return logits


def build_margin_loss(cfg):
    loss_name = getattr(cfg, "loss", "arcface").lower()
    scale = float(getattr(cfg, "scale", 64.0))

    if loss_name == "arcface":
        return CombinedMarginLoss(
            scale,
            cfg.margin_list[0],
            cfg.margin_list[1],
            cfg.margin_list[2],
            cfg.interclass_filtering_threshold,
        )
    if loss_name == "adaface":
        return AdaFace(
            s=scale,
            margin=float(getattr(cfg, "adaface_margin", 0.4)),
            h=float(getattr(cfg, "adaface_h", 0.333)),
            t_alpha=float(getattr(cfg, "adaface_t_alpha", 1.0)),
        )
    if loss_name == "cosface":
        return CosFace(
            s=scale,
            m=float(getattr(cfg, "cosface_margin", cfg.margin_list[2])),
        )

    raise ValueError(f"Unsupported loss '{loss_name}'. Use arcface, adaface, or cosface.")


"""Two-epoch, MS1MV3-only paired polish experiment; run through Slurm.

A: clipped original polish loss, layer3 causal scope.
B: A with all-site causal scope. C: B with fixed BN moments.
D: C with exact unclipped safe-row KD and early finite-prefix repair.
All arms use FP32; coefficients/radii are fixed. No IJB data enters training.
"""
from __future__ import annotations
import argparse
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from controlled_degree2 import losses
from controlled_degree2.augment import prepare_range_batch
from controlled_degree2.model import (quadratic_modules, load_controlled_checkpoint,
    load_teacher, set_quadratic_schedule, collect_range_stats,
    collect_causal_tail_penalty, operator_bound_targets, operator_bound_penalty,
    save_checkpoint)
from controlled_degree2.train import (attach_hints, hint_names, keep_batchnorm_eval,
    CausalTailReplay, load_canaries, evaluate_canaries)
from utils.utils_optimizer import clip_grad_norm_stable
from eval.finite_audit import FiniteAudit


def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()


def write(path, value):
    def clean(x):
        if isinstance(x,float) and not math.isfinite(x): return str(x)
        if isinstance(x,dict): return {str(k):clean(v) for k,v in x.items()}
        if isinstance(x,(list,tuple)): return [clean(v) for v in x]
        return x
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(clean(value),indent=2,allow_nan=False)+'\n');tmp.replace(path)


class Rows(Dataset):
    def __init__(self,root,rows,deterministic=False):
        self.root,self.rows,self.deterministic=root,rows,deterministic
        self.source=None
    def __len__(self): return len(self.rows)
    def __getitem__(self,i):
        if self.source is None:
            from dataset import MXFaceDataset
            self.source=MXFaceDataset(self.root,local_rank=0)
        row=int(self.rows[i])
        x,y=(self.source.get_oriented(row,0) if self.deterministic else self.source[row])
        return x,int(y),row


def seed_worker(worker_id):
    seed=torch.initial_seed()%2**32
    random.seed(seed);np.random.seed(seed)


class PrefixStop(Exception):
    def __init__(self,x): self.x=x


class PolishObjective(nn.Module):
    def __init__(self,student,teacher,arm,guard=4.):
        super().__init__()
        self.student,self.teacher,self.arm,self.guard=student,teacher,arm,guard
        self.teacher.eval().requires_grad_(False)
        self.sites=list(quadratic_modules(student))
        self.names=hint_names(student)
        self.sh,self.hs=attach_hints(student,self.names)
        self.th,self.ht=attach_hints(teacher,self.names)
        self.causal=[q.name for q in self.sites if arm!='A' or q.name.startswith('layer3.')]
        self.targets=operator_bound_targets(student,.1)
        self.metrics={};self.tail_scores=None

    def configure(self):
        self.student.train();self.teacher.eval()
        set_quadratic_schedule(self.student,alpha=1.,clip=self.arm!='D',clip_eval=False,
                               penalty='hinge',causal_guard_ratio=1.)
        if self.arm in ('C','D'): keep_batchnorm_eval(self.student)

    @torch.no_grad()
    def routes(self,images):
        """No-grad deployment probe. Invalid probe outputs never enter autograd."""
        self.student.eval()
        first=torch.full((len(images),),-1,dtype=torch.long,device=images.device)
        peak=torch.zeros(len(images),device=images.device)
        handles=[]
        for i,q in enumerate(self.sites):
            def observe(module,inputs,i=i):
                ratio=(inputs[0].abs()/module.lam_fit[None,:,None,None]).flatten(1).amax(1)
                bad=(~torch.isfinite(ratio)) | (ratio>self.guard)
                first[(first<0)&bad]=i
                peak.copy_(torch.maximum(peak,torch.nan_to_num(ratio,nan=float('inf'),posinf=float('inf'))))
            handles.append(q.register_forward_pre_hook(observe))
        try:
            output=self.student(images)
            if ((first<0)&~torch.isfinite(output).all(1)).any():
                raise FloatingPointError('output failed before any routable polynomial input; inspect checkpoint')
        finally:
            for h in handles:h.remove()
            self.sh.clear();self.configure()
        return first,peak

    def ordinary(self,images,mask,hint_weight,beta):
        self.sh.clear();self.th.clear()
        with torch.no_grad(): target=self.teacher(images)
        output=self.student(images)
        if not torch.isfinite(output).all() or not torch.isfinite(target).all():
            raise FloatingPointError('nonfinite ordinary/safe-row embedding')
        emb=losses.embedding_loss(output,target,mask)
        hint=losses.hint_loss(self.sh,self.th,self.names,mask) if mask.any() else output.sum()*0
        boundary,_,_=collect_range_stats(self.student)
        causal=collect_causal_tail_penalty(self.student,self.causal,guard_ratio=1.)
        value=emb+hint_weight*hint+beta*boundary+causal
        self.tail_scores=torch.stack([q.last_sample_ratio for q in self.sites if q.name in self.causal],1).amax(1).detach()
        return value,dict(cosine=float(emb.detach()),hint=float(hint.detach()),range=float(boundary.detach()),causal=float(causal.detach()))

    def repair(self,images,index):
        """Stop before the selected square; backprop only through its finite input."""
        q=self.sites[index]
        def stop(module,inputs): raise PrefixStop(inputs[0])
        handle=q.register_forward_pre_hook(stop)
        self.sh.clear()
        try:
            self.student(images)
        except PrefixStop as e:
            if not torch.isfinite(e.x).all():
                raise FloatingPointError('nonfinite prefix input: cannot repair this route')
            # FP64 reduction prevents overflow in the loss; forward backbone is FP32.
            relative=e.x.double().abs()/q.lam_fit.double()[None,:,None,None]
            excess=(relative-1.).clamp_min(0).square().flatten(1)
            return (excess.mean(1)+.01*excess.amax(1)).mean()
        finally: handle.remove()
        raise RuntimeError('repair boundary was not reached')

    def forward(self,images,mask,hint_weight,beta):
        self.configure()
        if self.arm!='D':
            loss,stats=self.ordinary(images,mask,hint_weight,beta)
            stats.update(safe_rows=len(images),repair_rows=0)
        else:
            routes,peak=self.routes(images)
            safe=routes<0;total=len(images)
            loss=images.new_zeros(())
            stats=dict(cosine=0.,hint=0.,range=0.,causal=0.,safe_rows=int(safe.sum()),repair_rows=int((~safe).sum()))
            if safe.any():
                normal,parts=self.ordinary(images[safe],mask[safe],hint_weight,beta)
                loss=loss+normal*safe.sum()/total;stats.update(parts)
            repair_loss=images.new_zeros(())
            for index in routes[~safe].unique().tolist():
                selected=routes==index
                repair_loss=repair_loss+self.repair(images[selected],index)*selected.sum()/total
            loss=loss+repair_loss
            stats['prefix']=float(repair_loss.detach())
            self.tail_scores=peak.detach()
        bound=operator_bound_penalty(self.student,self.targets)
        loss=loss+.0001*bound
        stats['bound']=float(bound.detach());stats['loss']=float(loss.detach())
        self.metrics=stats
        # Hooks otherwise retain unused prefix graphs between optimizer steps.
        self.sh.clear();self.th.clear()
        return loss


@torch.no_grad()
def audit(model,root,rows,workers,batch):
    """Fixed excluded MS1MV3 rows, original+flip; record all module boundaries."""
    model.eval();set_quadratic_schedule(model,alpha=1.,clip_eval=False)
    loader=DataLoader(Rows(root,rows,True),batch_size=batch,num_workers=workers,
                      multiprocessing_context='spawn' if workers else None)
    observer=FiniteAudit(model);bad=[];count=0;ratios={}
    def observe(q,inputs):
        r=(inputs[0].abs()/q.lam_fit[None,:,None,None]).flatten(1).amax(1)
        val=float(r.max());ratios[q.name]=max(ratios.get(q.name,0.),val) if math.isfinite(val) else float('inf')
    handles=[q.register_forward_pre_hook(observe) for q in quadratic_modules(model)]
    try:
        for images,_,ids in loader:
            for orientation in (0,1):
                x=images.cuda();x=x.flip(-1) if orientation else x
                output=model(x);finite=torch.isfinite(output).all(1)
                bad.extend([[int(ids[i]),orientation] for i in (~finite).nonzero().flatten().tolist()]);count+=len(x)
        result=observer.result()
    finally:
        observer.close()
        for h in handles:h.remove()
    result.update(source_images=len(rows),augmented_rows=count,embedding_nonfinite_rows=len(bad),
                  failures=bad,input_to_fit_radius_max=ratios,scope='fixed MS1MV3 subset excluded from this fine-tuning; teacher and initialization may have seen it')
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--arm',choices=list('ABCD'),required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--checkpoint',default='work_dirs/controlled_degree2_tail_ms1mv3_20260907/progressive/student_final.pt')
    p.add_argument('--teacher',default='work_dirs/ms1mv3_r50/model.pt')
    p.add_argument('--dataset-root',default='ms1m-retinaface-t1')
    p.add_argument('--epochs',type=int,default=2);p.add_argument('--batch-size',type=int,default=128)
    p.add_argument('--global-batch',type=int,default=2048);p.add_argument('--workers',type=int,default=4)
    p.add_argument('--seed',type=int,default=20260915);p.add_argument('--audit-images',type=int,default=8192)
    p.add_argument('--tail-manifest',default=None)
    p.add_argument('--smoke',action='store_true');args=p.parse_args()
    if args.epochs<1 or args.audit_images<1: p.error('positive epochs/audit size required')
    rank=int(os.environ.get('RANK',0));world=int(os.environ.get('WORLD_SIZE',1));local=int(os.environ.get('LOCAL_RANK',0))
    torch.cuda.set_device(local);torch.set_num_threads(2)
    if world>1:torch.distributed.init_process_group('nccl')
    random.seed(args.seed+rank);np.random.seed(args.seed+rank);torch.manual_seed(args.seed+rank)
    root=Path(args.output)
    if rank==0:root.mkdir(parents=True,exist_ok=False)
    if world>1:torch.distributed.barrier()
    if args.global_batch%(args.batch_size*world):raise ValueError('global batch must divide microbatch*world')
    accum=args.global_batch//(args.batch_size*world)
    from dataset import MXFaceDataset
    ds=MXFaceDataset(args.dataset_root,local_rank=local);n=len(ds);del ds
    rng=np.random.default_rng(args.seed);eval_rows=np.sort(rng.choice(n,args.audit_images,replace=False))
    train_rows=np.setdiff1d(np.arange(n),eval_rows)
    dataset=Rows(args.dataset_root,train_rows)
    sampler=DistributedSampler(dataset,num_replicas=world,rank=rank,seed=args.seed,drop_last=True)
    generator=torch.Generator().manual_seed(args.seed+rank)
    loader=DataLoader(dataset,batch_size=args.batch_size,sampler=sampler,drop_last=True,num_workers=args.workers,
                      multiprocessing_context='spawn' if args.workers else None,worker_init_fn=seed_worker,generator=generator)
    student,payload=load_controlled_checkpoint(args.checkpoint,'cuda')
    teacher=load_teacher(args.teacher,'cuda').eval().requires_grad_(False)
    frozen={k:v.detach().cpu().clone() for k,v in student.state_dict().items() if any(k.endswith(s) for s in ('coeffs','lam_fit','lam_reg','running_mean','running_var','num_batches_tracked'))}
    obj=PolishObjective(student,teacher,args.arm)
    wrapped=DDP(obj,device_ids=[local],find_unused_parameters=True,broadcast_buffers=True) if world>1 else obj
    params=[v for v in obj.parameters() if v.requires_grad]
    optim=torch.optim.SGD(params,lr=.002,momentum=.9,nesterov=True,weight_decay=.0005)
    steps_per_epoch=2 if args.smoke else len(loader)//accum
    schedule_steps_per_epoch=len(loader)//accum
    total_steps=(2 if args.smoke else args.epochs)*schedule_steps_per_epoch
    config=vars(args)|dict(world_size=world,accumulation=accum,steps_per_epoch=steps_per_epoch,precision='fp32',
            base_lr=.002,causal_sites=obj.causal,bn_moments_fixed=args.arm in ('C','D'),coefficients_fixed=True,
            approximation_target='original per-channel PReLU',interval='checkpoint [-lam_fit,lam_fit]; no interval change',
            checkpoint_sha256=digest(args.checkpoint),teacher_sha256=digest(args.teacher),
            data_selection='MS1MV3 only; no IJB fitting or selection',inference_clipping=False,
            routing_guard_fit_ratio=4.,repair_target_fit_ratio=1.,train_rows=len(train_rows),
            eval_rows_sha256=hashlib.sha256(eval_rows.tobytes()).hexdigest())
    if rank==0:
        write(root/'config.json',config);np.save(root/'audit_rows.npy',eval_rows)
    canaries=load_canaries(args.dataset_root,['lfw']) if rank==0 and not args.smoke else None
    tail_rows=None
    if args.tail_manifest:
        manifest=json.loads(Path(args.tail_manifest).read_text())
        if Path(manifest['checkpoint']).resolve() != Path(args.checkpoint).resolve():
            raise ValueError('tail baseline checkpoint mismatch')
        rows=set(int(r['source_index']) for r in manifest['output_nonfinite'])
        for data in manifest['activations'].values():
            rows.update(int(r['source_index']) for r in data['tail'])
        tail_rows=np.array(sorted(rows),dtype=np.int64)
    records=[];step=0;replay=CausalTailReplay(1024,.25,'cuda');start=time.time()
    def evaluate(epoch):
        if world>1:torch.distributed.barrier()
        if rank==0:
            # Evaluation must not consume training augmentation RNG or alter BN buffers.
            py=random.getstate();npstate=np.random.get_state()
            with torch.random.fork_rng(devices=[local]):
                probe=audit(student,args.dataset_root,eval_rows,args.workers,32 if args.smoke else 128)
                canary=evaluate_canaries(student,canaries,256) if canaries else None
                tails=audit(student,args.dataset_root,tail_rows,args.workers,128) if tail_rows is not None and len(tail_rows) else None
            random.setstate(py);np.random.set_state(npstate)
            write(root/f'audit_epoch{epoch}.json',probe)
            if tails is not None:
                tails['scope']='fixed baseline MS1MV3 training tails, report only; not held-out'
                write(root/f'tail_audit_epoch{epoch}.json',tails)
            records.append(dict(epoch=epoch,step=step,lfw=canary,nonfinite_rows=probe['embedding_nonfinite_rows'],nonfinite_values=probe['nonfinite_values'],baseline_tail_nonfinite_rows=None if tails is None else tails['embedding_nonfinite_rows']))
            write(root/'trend.json',records)
            print('EVALUATION',records[-1],flush=True)
        if world>1:torch.distributed.barrier()
    evaluate(0)
    optim.zero_grad(set_to_none=True)
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch);epoch_stats=dict(safe_rows=0,repair_rows=0)
        for batch_index,(images,_,_) in enumerate(loader):
            if batch_index>=steps_per_epoch*accum:break
            images=images.cuda()
            images,mask=prepare_range_batch(images,pathological_fraction=.05,crop_probability=.1,
                          lowres_probability=.2,photo_probability=.2,stress_probability=.4)
            if step>=100:images=replay.mix(images)
            sync=((batch_index+1)%accum==0)
            ctx=wrapped.no_sync() if world>1 and not sync else contextlib.nullcontext()
            with ctx:
                value=wrapped(images,mask,losses.hint_weight_at(step,total_steps,1.,.3),
                              losses.beta_at(step,round(.1*schedule_steps_per_epoch),1.))
                flag=torch.tensor(int(torch.isfinite(value)),device='cuda')
                if world>1:torch.distributed.all_reduce(flag,op=torch.distributed.ReduceOp.MIN)
                if not flag:raise FloatingPointError('nonfinite loss; no update applied')
                (value/accum).backward()
            for k in epoch_stats:epoch_stats[k]+=obj.metrics[k]
            if not sync:continue
            flag=torch.tensor(int(all(v.grad is None or torch.isfinite(v.grad).all() for v in params)),device='cuda')
            if world>1:torch.distributed.all_reduce(flag,op=torch.distributed.ReduceOp.MIN)
            if not flag:raise FloatingPointError('nonfinite gradient; no update applied')
            norm=clip_grad_norm_stable(params,5.,error_if_nonfinite=True)
            lr=.002*losses.lr_factor(step,round(.5*schedule_steps_per_epoch),total_steps)
            for g in optim.param_groups:g['lr']=lr
            optim.step();optim.zero_grad(set_to_none=True);step+=1
            if mask.any() and obj.tail_scores is not None:replay.update(images[mask],obj.tail_scores[mask])
            if rank==0 and (step%25==0 or args.smoke):
                print(json.dumps(dict(step=step,lr=lr,grad=float(norm),**obj.metrics)),flush=True)
        for k,v in frozen.items():
            if args.arm in ('C','D') or any(k.endswith(s) for s in ('coeffs','lam_fit','lam_reg')):
                if not torch.equal(student.state_dict()[k].cpu(),v):raise RuntimeError('fixed tensor changed: '+k)
        counts=torch.tensor(list(epoch_stats.values()),device='cuda',dtype=torch.long)
        if world>1:torch.distributed.all_reduce(counts)
        if rank==0:
            set_quadratic_schedule(student,alpha=1.,clip_eval=False)
            save_checkpoint(str(root/f'epoch{epoch+1}.pt'),student,payload['poly_calib'],teacher_weights=args.teacher,
                            extra=dict(epoch=epoch,step=step,train_config=config,origin='polish_abcd',diagnostic_only=True))
            write(root/f'train_epoch{epoch+1}.json',dict(zip(epoch_stats,counts.tolist()))|dict(step=step,elapsed_seconds=time.time()-start))
        evaluate(epoch+1)
    if args.smoke and args.arm=='D':
        # Exercise the all-prefix DDP/unused-parameter path without an update.
        obj.guard=.001
        optim.zero_grad(set_to_none=True)
        routed=wrapped(images,mask,.3,1.)
        routed.backward()
        ok=all(v.grad is None or torch.isfinite(v.grad).all() for v in params)
        flag=torch.tensor(int(ok and obj.metrics['repair_rows']>0),device='cuda')
        if world>1:torch.distributed.all_reduce(flag,op=torch.distributed.ReduceOp.MIN)
        if not flag:raise RuntimeError('forced prefix-routing smoke failed')
        if rank==0:write(root/'routing_smoke.json',dict(finite_gradients=True,optimizer_update=False,**obj.metrics))
        optim.zero_grad(set_to_none=True);obj.guard=4.
    if rank==0:
        write(root/'completed.json',dict(arm=args.arm,steps=step,epochs=args.epochs,smoke=args.smoke,finite_updates=True))
    if world>1:torch.distributed.destroy_process_group()


if __name__=='__main__':main()

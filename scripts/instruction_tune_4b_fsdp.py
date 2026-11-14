"""FSDP supervised instruction tuning for the pretrained Neuron Forge 4B checkpoint."""
from __future__ import annotations
import argparse, json, math, os, shutil
from functools import partial
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import get_model_state_dict, set_model_state_dict, get_state_dict, set_state_dict
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from neuronforge_engine.config import load_config
from neuronforge_engine.model import Block, CausalSelfAttention, GPT

EXPECTED_PARAMETERS=4_064_498_688
MINIMUM_GPU_COUNT=8

class TrainingState(Stateful):
    def __init__(self,model:nn.Module,optimizer:torch.optim.Optimizer): self.model,self.optimizer=model,optimizer
    def state_dict(self):
        m,o=get_state_dict(self.model,self.optimizer); return {"model":m,"optimizer":o}
    def load_state_dict(self,state):
        set_state_dict(self.model,self.optimizer,model_state_dict=state["model"],optim_state_dict=state["optimizer"])

class ModelOnlyState(Stateful):
    """Load pretrained model weights without importing pretraining AdamW moments."""
    def __init__(self, model: nn.Module):
        self.model = model

    def state_dict(self):
        return {"model": get_model_state_dict(self.model)}

    def load_state_dict(self, state):
        set_model_state_dict(self.model, model_state_dict=state["model"])


def init_meta(module,device):
    module.to_empty(device=device,recurse=False)
    for p in module.parameters(recurse=False):
        if p.ndim>=2: init.normal_(p,mean=0.0,std=0.02)
        else: init.ones_(p)
    for name,b in module.named_buffers(recurse=False):
        if name=="inv_freq" and isinstance(module,CausalSelfAttention):
            d=b.numel()*2; b.copy_(1.0/(10000**(torch.arange(0,d,2,device=device,dtype=torch.float32)/d)))
        elif b.is_floating_point(): b.zero_()

def build_model(config,device):
    with torch.device("meta"): model=GPT(config.model)
    model.gradient_checkpointing=bool(config.train.gradient_checkpointing)
    return FSDP(model,auto_wrap_policy=partial(transformer_auto_wrap_policy,transformer_layer_cls={Block}),
      param_init_fn=lambda m:init_meta(m,device),sharding_strategy=ShardingStrategy.FULL_SHARD,
      mixed_precision=MixedPrecision(param_dtype=torch.bfloat16,reduce_dtype=torch.bfloat16,buffer_dtype=torch.bfloat16),
      device_id=device,use_orig_params=True,limit_all_gathers=True)

def shards(root,split,block):
    m=json.loads((root/"manifest.json").read_text())
    n=int(m[f"{split}_examples"])
    x=np.memmap(root/m["files"][f"{split}_inputs"],mode="r",dtype=np.uint16,shape=(n,block+1))
    y=np.memmap(root/m["files"][f"{split}_labels"],mode="r",dtype=np.int32,shape=(n,block+1))
    return x,y

def batch(xs,ys,size,rng,device):
    ix=rng.integers(0,len(xs),size=size)
    x=torch.from_numpy(np.asarray(xs[ix,:-1],dtype=np.int64).copy()).to(device)
    y=torch.from_numpy(np.asarray(ys[ix,1:],dtype=np.int64).copy()).to(device)
    return x,y

def loss_for(model,x,y):
    logits,_=model(x)
    return F.cross_entropy(logits.reshape(-1,logits.size(-1)),y.reshape(-1),ignore_index=-100)

@torch.no_grad()
def evaluate(model,xs,ys,cfg,rng,device,world):
    model.eval(); total=torch.zeros((),dtype=torch.float64,device=device)
    for _ in range(cfg.train.eval_batches):
        x,y=batch(xs,ys,cfg.train.batch_size,rng,device)
        with torch.autocast(device_type="cuda",dtype=torch.bfloat16): loss=loss_for(model,x,y)
        total+=loss.double()
    total/=cfg.train.eval_batches; dist.all_reduce(total); total/=world
    model.train(); return float(total.item())

def save(model,opt,root,step,best,rank):
    folder=root/f"step-{step:09d}"
    if rank==0: folder.mkdir(parents=True,exist_ok=True)
    dist.barrier(); dcp.save({"training":TrainingState(model,opt)},checkpoint_id=str(folder)); dist.barrier()
    if rank==0:
        meta={"step":step,"best_validation_loss":best}
        (folder/"run-state.json").write_text(json.dumps(meta,indent=2)+"\n")
        tmp=root/"latest.json.tmp"; tmp.write_text(json.dumps({"checkpoint":folder.name,**meta},indent=2)+"\n"); tmp.replace(root/"latest.json")
        old=sorted(p for p in root.glob("step-*") if p.is_dir() and p!=folder)
        for p in old[:-1]: shutil.rmtree(p,ignore_errors=True)
        print(f"Saved resumable SFT checkpoint {folder}",flush=True)
    dist.barrier()

def restore(model,opt,root):
    p=root/"latest.json"
    if not p.exists(): return 0,float("inf")
    m=json.loads(p.read_text()); folder=root/m["checkpoint"]
    dcp.load({"training":TrainingState(model,opt)},checkpoint_id=str(folder)); dist.barrier()
    return int(m["step"]),float(m.get("best_validation_loss",float("inf")))

def append_metric(root: Path, record: dict, rank: int):
    if rank == 0:
        with (root / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config",type=Path,default=Path("configs/4b-sft.yaml"))
    ap.add_argument("--data-dir",type=Path,default=Path("data/4b-sft"))
    ap.add_argument("--pretrained-checkpoint",type=Path,default=Path("checkpoints/neuron-forge-4b"))
    ap.add_argument("--checkpoint-dir",type=Path,default=Path("checkpoints/neuron-forge-4b-sft"))
    ap.add_argument("--max-steps",type=int,default=0); ap.add_argument("--no-resume",action="store_true")
    a=ap.parse_args()
    if "LOCAL_RANK" not in os.environ: raise RuntimeError("Launch with torchrun.")
    dist.init_process_group("nccl"); rank=dist.get_rank(); world=dist.get_world_size()
    local=int(os.environ["LOCAL_RANK"]); torch.cuda.set_device(local); device=torch.device("cuda",local)
    try:
        if world<MINIMUM_GPU_COUNT: raise RuntimeError(f"Need at least {MINIMUM_GPU_COUNT} GPUs, got {world}.")
        if not torch.cuda.is_bf16_supported(): raise RuntimeError("BF16 GPUs required.")
        cfg=load_config(a.config)
        if cfg.model.parameter_count()!=EXPECTED_PARAMETERS: raise ValueError("SFT architecture differs from pretrained 4B model.")
        base_meta=json.loads((a.pretrained_checkpoint/"latest.json").read_text())
        base_folder=a.pretrained_checkpoint/base_meta["checkpoint"]
        manifest=json.loads((a.data_dir/"manifest.json").read_text())
        if manifest.get("format")!="neuronforge-sft-v1" or not manifest.get("assistant_only_loss"):
            raise ValueError("Missing or invalid assistant-only SFT data manifest.")
        tx,ty=shards(a.data_dir,"train",cfg.model.block_size); vx,vy=shards(a.data_dir,"validation",cfg.model.block_size)
        model=build_model(cfg,device)
        optimizer=torch.optim.AdamW(model.parameters(),lr=cfg.train.learning_rate,betas=(0.9,0.95),weight_decay=cfg.train.weight_decay)
        dcp.load({"training":ModelOnlyState(model)},checkpoint_id=str(base_folder))
        a.checkpoint_dir.mkdir(parents=True,exist_ok=True)
        start,best=(0,float("inf")) if a.no_resume else restore(model,optimizer,a.checkpoint_dir)
        steps=a.max_steps if a.max_steps>0 else cfg.train.max_steps
        rng=np.random.default_rng(cfg.train.seed+rank); model.train()
        for step in range(start,steps):
            if step<cfg.train.warmup_steps: lr=cfg.train.learning_rate*(step+1)/max(1,cfg.train.warmup_steps)
            else:
                progress=min(1.0,(step-cfg.train.warmup_steps)/max(1,steps-cfg.train.warmup_steps))
                lr=cfg.train.min_learning_rate+0.5*(cfg.train.learning_rate-cfg.train.min_learning_rate)*(1+math.cos(math.pi*progress))
            for group in optimizer.param_groups: group["lr"]=lr
            optimizer.zero_grad(set_to_none=True); running=torch.zeros((),device=device)
            for _ in range(cfg.train.gradient_accumulation_steps):
                x,y=batch(tx,ty,cfg.train.batch_size,rng,device)
                with torch.autocast(device_type="cuda",dtype=torch.bfloat16):
                    loss=loss_for(model,x,y)/cfg.train.gradient_accumulation_steps
                loss.backward(); running+=loss.detach()
            model.clip_grad_norm_(cfg.train.grad_clip); optimizer.step(); done=step+1
            if done==1 or done%10==0:
                metric=running.float(); dist.all_reduce(metric); metric/=world
                if rank == 0:
                    print(f"SFT step {done}/{steps} | assistant_loss {metric.item():.4f} | lr {lr:.3e}",flush=True)
                    append_metric(a.checkpoint_dir,{"stage":"instruction_tuning","step":done,"assistant_train_loss":float(metric.item()),"learning_rate":lr},rank)
            if done%cfg.train.eval_interval==0 or done==steps:
                val=evaluate(model,vx,vy,cfg,rng,device,world); best=min(best,val)
                if rank==0:
                    print(f"SFT validation step {done} | loss {val:.4f} | best {best:.4f}",flush=True)
                    append_metric(a.checkpoint_dir,{"stage":"instruction_tuning","step":done,"validation_loss":val,"best_validation_loss":best},rank)
                save(model,optimizer,a.checkpoint_dir,done,best,rank)
        if rank==0: print("Instruction tuning finished. Run conversational evaluation before release.",flush=True)
    finally:
        if dist.is_initialized(): dist.destroy_process_group()

if __name__=="__main__": main()

"""Distributed 4B-class pretraining with PyTorch FSDP and sharded checkpoints.

Launch with torchrun on a multi-GPU CUDA host. This script deliberately refuses
to run on a single GPU or CPU; it is not a Colab T4 workload.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.init as init
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

from neuronforge_engine.config import load_config
from neuronforge_engine.model import Block, CausalSelfAttention, GPT, RMSNorm


EXPECTED_PARAMETERS = 4_064_498_688
MINIMUM_GPU_COUNT = 8


class TrainingState(Stateful):
    """FSDP-aware model and optimizer state for Distributed Checkpoint."""

    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer):
        self.model = model
        self.optimizer = optimizer

    def state_dict(self):
        model_state, optimizer_state = get_state_dict(self.model, self.optimizer)
        return {"model": model_state, "optimizer": optimizer_state}

    def load_state_dict(self, state_dict):
        set_state_dict(
            self.model,
            self.optimizer,
            model_state_dict=state_dict["model"],
            optim_state_dict=state_dict["optimizer"],
        )


def initialize_meta_module(module: nn.Module, device: torch.device) -> None:
    """Materialize only this module's own meta tensors before FSDP shards them."""
    module.to_empty(device=device, recurse=False)
    for parameter in module.parameters(recurse=False):
        if parameter.ndim >= 2:
            init.normal_(parameter, mean=0.0, std=0.02)
        else:
            init.ones_(parameter)
    for name, buffer in module.named_buffers(recurse=False):
        if name == "inv_freq" and isinstance(module, CausalSelfAttention):
            head_dim = buffer.numel() * 2
            values = 1.0 / (
                10000 ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
            )
            buffer.copy_(values)
        elif buffer.is_floating_point():
            buffer.zero_()


def create_fsdp_model(config, device: torch.device) -> FSDP:
    # Constructing on meta avoids allocating a full 4B copy on every rank.
    with torch.device("meta"):
        model = GPT(config.model)
    model.gradient_checkpointing = bool(config.train.gradient_checkpointing)

    auto_wrap = partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={Block},
    )
    precision = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )
    return FSDP(
        model,
        auto_wrap_policy=auto_wrap,
        param_init_fn=lambda module: initialize_meta_module(module, device),
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=precision,
        device_id=device,
        use_orig_params=True,
        limit_all_gathers=True,
    )


def get_batch(data: np.memmap, batch_size: int, block_size: int, rng, device):
    maximum_start = len(data) - block_size - 1
    if maximum_start < 1:
        raise ValueError(f"Token shard contains {len(data):,} tokens, too few for block size {block_size}.")
    starts = rng.integers(0, maximum_start, size=batch_size)
    rows = np.stack([data[int(start):int(start) + block_size + 1] for start in starts])
    batch = torch.from_numpy(rows.astype(np.int64, copy=False)).to(device, non_blocking=True)
    return batch[:, :-1], batch[:, 1:]


def learning_rate(step: int, total_steps: int, config) -> float:
    if config.train.warmup_steps and step < config.train.warmup_steps:
        return config.train.learning_rate * (step + 1) / config.train.warmup_steps
    decay_steps = max(1, total_steps - config.train.warmup_steps)
    progress = min(1.0, max(0.0, (step - config.train.warmup_steps) / decay_steps))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return config.train.min_learning_rate + cosine * (
        config.train.learning_rate - config.train.min_learning_rate
    )


@torch.no_grad()
def evaluate(model, validation_data, config, rng, device, world_size):
    model.eval()
    total = torch.zeros((), dtype=torch.float64, device=device)
    for _ in range(config.train.eval_batches):
        x, y = get_batch(
            validation_data,
            config.train.batch_size,
            config.model.block_size,
            rng,
            device,
        )
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, loss = model(x, y)
        total += loss.detach().double()
    total /= config.train.eval_batches
    dist.all_reduce(total, op=dist.ReduceOp.SUM)
    total /= world_size
    model.train()
    return float(total.item())


def save_checkpoint(model, optimizer, checkpoint_root: Path, step: int, best_loss: float, rank: int):
    checkpoint_dir = checkpoint_root / f"step-{step:09d}"
    if rank == 0:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier()
    state = TrainingState(model, optimizer)
    dcp.save({"training": state}, checkpoint_id=str(checkpoint_dir))
    dist.barrier()
    if rank == 0:
        metadata = {"step": step, "best_validation_loss": best_loss}
        (checkpoint_dir / "run-state.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        temporary = checkpoint_root / "latest.json.tmp"
        temporary.write_text(json.dumps({"checkpoint": checkpoint_dir.name, **metadata}, indent=2) + "\n", encoding="utf-8")
        temporary.replace(checkpoint_root / "latest.json")
        old_checkpoints = sorted(
            path for path in checkpoint_root.glob("step-*") if path.is_dir() and path != checkpoint_dir
        )
        for old in old_checkpoints[:-1]:
            shutil.rmtree(old, ignore_errors=True)
        print(f"Saved distributed checkpoint: {checkpoint_dir}", flush=True)
    dist.barrier()


def load_checkpoint(model, optimizer, checkpoint_root: Path, rank: int):
    latest_path = checkpoint_root / "latest.json"
    if not latest_path.is_file():
        return 0, float("inf")
    metadata = json.loads(latest_path.read_text(encoding="utf-8"))
    checkpoint_dir = checkpoint_root / metadata["checkpoint"]
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Latest checkpoint directory is missing: {checkpoint_dir}")
    dcp.load(
        {"training": TrainingState(model, optimizer)},
        checkpoint_id=str(checkpoint_dir),
    )
    dist.barrier()
    if rank == 0:
        print(f"Resumed distributed checkpoint at step {metadata['step']}", flush=True)
    return int(metadata["step"]), float(metadata.get("best_validation_loss", float("inf")))


def append_metric(root: Path, record: dict, rank: int) -> None:
    if rank == 0:
        with (root / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/4b.yaml"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/4b"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/neuron-forge-4b"))
    parser.add_argument("--max-steps", type=int, default=0, help="Optional short-run override; 0 uses target_tokens.")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--allow-small-corpus", action="store_true", help="Allow an intentionally tiny corpus for a short infrastructure smoke test only.")
    args = parser.parse_args()

    if "LOCAL_RANK" not in os.environ:
        raise RuntimeError("Launch this program with torchrun; direct python execution is not supported.")
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    try:
        if world_size < MINIMUM_GPU_COUNT:
            raise RuntimeError(
                f"4B training requires at least {MINIMUM_GPU_COUNT} GPUs for this FSDP setup; "
                f"torchrun provided {world_size}."
            )
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("All training GPUs must support BF16.")
        config = load_config(args.config)
        actual_parameters = config.model.parameter_count()
        if actual_parameters != EXPECTED_PARAMETERS:
            raise ValueError(f"4B architecture mismatch: expected {EXPECTED_PARAMETERS:,}, got {actual_parameters:,}")

        train_path = args.data_dir / "train.bin"
        validation_path = args.data_dir / "validation.bin"
        tokenizer_path = args.data_dir / "tokenizer.json"
        for path in (train_path, validation_path, tokenizer_path):
            if not path.is_file():
                raise FileNotFoundError(f"Prepared 4B data artifact missing: {path}. Run scripts/prepare_4b_data.py first.")
        train_data = np.memmap(train_path, mode="r", dtype=np.uint16)
        validation_data = np.memmap(validation_path, mode="r", dtype=np.uint16)
        if min(len(train_data), len(validation_data)) <= config.model.block_size + 1:
            raise ValueError("Prepared token shards are too small for the configured context length.")
        required_tokens = int(config.train.minimum_data_tokens)
        if len(train_data) < required_tokens and not args.allow_small_corpus:
            raise RuntimeError(
                f"Refusing full 4B pretraining: corpus has {len(train_data):,} training tokens, "
                f"but this run requires at least {required_tokens:,}. The starter Wikipedia subset is not enough. "
                "Enable additional license-reviewed data sources and prepare a larger, filtered corpus. "
                "For an explicitly limited infrastructure test only, pass --allow-small-corpus."
            )
        if len(train_data) < required_tokens and rank == 0:
            print("WARNING: small-corpus override enabled; this run is only an infrastructure smoke test, not useful 4B pretraining.", flush=True)

        tokens_per_step = (
            config.train.batch_size
            * config.model.block_size
            * config.train.gradient_accumulation_steps
            * world_size
        )
        total_steps = (
            args.max_steps
            if args.max_steps > 0
            else math.ceil(config.train.target_tokens / tokens_per_step)
            if config.train.target_tokens > 0
            else config.train.max_steps
        )
        if total_steps < 1:
            raise ValueError("Total training steps must be positive.")

        seed = int(config.train.seed)
        random.seed(seed)
        np.random.seed(seed + rank)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        rng = np.random.default_rng(seed + rank)

        if rank == 0:
            print(
                f"Neuron Forge 4B | {actual_parameters:,} parameters | {world_size} GPUs | "
                f"{tokens_per_step:,} tokens/update | target {total_steps:,} updates "
                f"({total_steps * tokens_per_step:,} tokens)",
                flush=True,
            )
            print(f"Training token shard: {len(train_data):,} | validation: {len(validation_data):,}", flush=True)

        model = create_fsdp_model(config, device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.train.learning_rate,
            betas=(0.9, 0.95),
            weight_decay=config.train.weight_decay,
        )
        args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        if args.no_resume:
            start_step, best_loss = 0, float("inf")
        else:
            start_step, best_loss = load_checkpoint(model, optimizer, args.checkpoint_dir, rank)

        if start_step >= total_steps:
            if rank == 0:
                print(f"Training already reached target step {total_steps:,}; no further updates required.")
            return

        model.train()
        for step in range(start_step, total_steps):
            lr = learning_rate(step, total_steps, config)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            running_loss = torch.zeros((), dtype=torch.float32, device=device)

            for _ in range(config.train.gradient_accumulation_steps):
                x, y = get_batch(
                    train_data,
                    config.train.batch_size,
                    config.model.block_size,
                    rng,
                    device,
                )
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    _, loss = model(x, y)
                    scaled_loss = loss / config.train.gradient_accumulation_steps
                scaled_loss.backward()
                running_loss += loss.detach().float()

            model.clip_grad_norm_(config.train.grad_clip)
            optimizer.step()
            completed_step = step + 1

            if completed_step % 10 == 0 or completed_step == 1:
                averaged = running_loss / config.train.gradient_accumulation_steps
                dist.all_reduce(averaged, op=dist.ReduceOp.SUM)
                averaged /= world_size
                if rank == 0:
                    print(
                        f"step {completed_step:,}/{total_steps:,} | loss {averaged.item():.4f} "
                        f"| lr {lr:.3e} | tokens seen {completed_step * tokens_per_step:,}",
                        flush=True,
                    )
                    append_metric(args.checkpoint_dir, {"stage": "pretraining", "step": completed_step, "train_loss": float(averaged.item()), "learning_rate": lr, "tokens_seen": completed_step * tokens_per_step}, rank)

            should_evaluate = completed_step % config.train.eval_interval == 0 or completed_step == total_steps
            if should_evaluate:
                validation_loss = evaluate(model, validation_data, config, rng, device, world_size)
                if validation_loss < best_loss:
                    best_loss = validation_loss
                if rank == 0:
                    print(
                        f"evaluation step {completed_step:,} | validation loss {validation_loss:.4f} "
                        f"| best {best_loss:.4f}",
                        flush=True,
                    )
                    append_metric(args.checkpoint_dir, {"stage": "pretraining", "step": completed_step, "validation_loss": validation_loss, "best_validation_loss": best_loss}, rank)
                save_checkpoint(model, optimizer, args.checkpoint_dir, completed_step, best_loss, rank)

        if rank == 0:
            print(
                f"Training target reached: {total_steps:,} optimizer updates; "
                f"{total_steps * tokens_per_step:,} training tokens processed.",
                flush=True,
            )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()

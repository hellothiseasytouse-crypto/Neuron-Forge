"""End-to-end training loop for a randomly initialized Neuron Forge GPT."""
import math
import random
from pathlib import Path

import torch

from .checkpoint import read_checkpoint, save_checkpoint
from .config import Config
from .data import encode_and_split, get_batch, load_text
from .model import GPT
from .tokenizer import ByteBPETokenizer


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no CUDA-enabled PyTorch device is available")
    if device.type == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        raise RuntimeError("MPS was requested but is not available on this machine")
    return device


def _optimizer(model: GPT, cfg):
    decay, no_decay = [], []
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        (decay if parameter.dim() >= 2 else no_decay).append(parameter)
    return torch.optim.AdamW([
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ], lr=cfg.learning_rate, betas=(0.9, 0.95))


@torch.no_grad()
def estimate_loss(model: GPT, train_data: torch.Tensor, val_data: torch.Tensor,
                  block_size: int, batch_size: int, eval_batches: int, device):
    model.eval()
    results = {}
    for name, data in (("train", train_data), ("val", val_data)):
        losses = []
        for _ in range(eval_batches):
            x, y = get_batch(data, block_size, batch_size, device)
            _, loss = model(x, y)
            losses.append(float(loss.item()))
        results[name] = sum(losses) / len(losses)
    model.train()
    return results


def learning_rate_at(step: int, cfg) -> float:
    if cfg.warmup_steps and step < cfg.warmup_steps:
        return cfg.learning_rate * (step + 1) / cfg.warmup_steps
    decay_steps = max(1, cfg.max_steps - cfg.warmup_steps)
    progress = min(1.0, max(0.0, (step - cfg.warmup_steps) / decay_steps))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.min_learning_rate + cosine * (cfg.learning_rate - cfg.min_learning_rate)


def train(config: Config, resume: bool = False) -> Path:
    config.validate()
    random.seed(config.train.seed)
    torch.manual_seed(config.train.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.train.seed)
    device = choose_device(config.train.device)
    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_amp and getattr(torch.cuda, "is_bf16_supported", lambda: False)() else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and amp_dtype == torch.float16)
    precision_label = "bfloat16" if use_amp and amp_dtype == torch.bfloat16 else ("float16" if use_amp else "float32")
    print(f"Neuron Forge training device: {device} | precision: {precision_label}")

    text = load_text(config.train.data_path)
    checkpoint_path = Path(config.train.checkpoint_path)
    saved = None
    if resume:
        saved = read_checkpoint(checkpoint_path, device="cpu")
        if saved.get("format_version") not in {2, 3}:
            raise ValueError("This checkpoint format cannot be resumed. Start a new training run.")
        tokenizer = ByteBPETokenizer.from_state_dict(saved["tokenizer"])
        print(f"Resuming from step {saved.get('step', 0)}")
    else:
        tokenizer = ByteBPETokenizer.train(
            text,
            vocab_size=config.tokenizer.vocab_size,
            min_frequency=config.tokenizer.min_frequency,
            max_train_chars=config.tokenizer.max_train_chars,
        )
    if tokenizer.vocab_size != config.tokenizer.vocab_size:
        raise ValueError(
            f"Tokenizer learned {tokenizer.vocab_size:,} of {config.tokenizer.vocab_size:,} requested tokens. "
            "Add more diverse corpus text before training this architecture."
        )
    config.model.vocab_size = tokenizer.vocab_size
    train_data, val_data = encode_and_split(text, tokenizer)
    needed = config.model.block_size + 1
    if len(train_data) <= needed or len(val_data) <= needed:
        raise ValueError(
            f"Corpus is too small for block_size={config.model.block_size}. "
            "Add a larger corpus or lower model.block_size."
        )

    model = GPT(config.model).to(device)
    model.gradient_checkpointing = bool(config.train.gradient_checkpointing)
    if model.gradient_checkpointing:
        print("Activation checkpointing: enabled")
    optimizer = _optimizer(model, config.train)
    start_step = 0
    best_val = float("inf")
    if saved is not None:
        if saved["model_config"] != config.to_dict()["model"]:
            raise ValueError("Resume checkpoint model configuration does not match the current config")
        model.load_state_dict(saved["model_state"])
        if saved.get("optimizer_state"):
            optimizer.load_state_dict(saved["optimizer_state"])
        start_step = int(saved.get("step", 0))
        best_val = float(saved.get("best_val_loss", float("inf")))

    print(f"Vocabulary: {tokenizer.vocab_size} | Parameters: {model.num_parameters():,}")
    print(f"Training tokens: {len(train_data):,} | Validation tokens: {len(val_data):,}")
    print("Weights initialized from scratch; no pretrained checkpoint is used.")
    model.train()
    for step in range(start_step, config.train.max_steps):
        lr = learning_rate_at(step, config.train)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        for _ in range(config.train.gradient_accumulation_steps):
            x, y = get_batch(train_data, config.model.block_size, config.train.batch_size, device)
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    _, loss = model(x, y)
            else:
                _, loss = model(x, y)
            scaler.scale(loss / config.train.gradient_accumulation_steps).backward()
        scaler.unscale_(optimizer)
        if config.train.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        completed_step = step + 1
        if completed_step == 1 or completed_step % config.train.eval_interval == 0 or completed_step == config.train.max_steps:
            metrics = estimate_loss(model, train_data, val_data, config.model.block_size,
                                    config.train.batch_size, config.train.eval_batches, device)
            print(f"step {completed_step:>6} | lr {lr:.2e} | train {metrics['train']:.4f} | val {metrics['val']:.4f}")
            if metrics["val"] < best_val:
                best_val = metrics["val"]
                best_path = checkpoint_path.with_name("best" + checkpoint_path.suffix)
                save_checkpoint(best_path, model, tokenizer, config.to_dict(), completed_step, best_val)

            # Keep a resumable checkpoint with optimizer state at every evaluation.
            # Cloud notebook runtimes can end without warning; the next session
            # should continue from the last fully saved optimizer update.
            save_checkpoint(checkpoint_path, model, tokenizer, config.to_dict(),
                            completed_step, best_val, optimizer=optimizer)
            print(f"Saved resumable checkpoint at step {completed_step}: {checkpoint_path}")

    save_checkpoint(checkpoint_path, model, tokenizer, config.to_dict(), config.train.max_steps,
                    best_val, optimizer=optimizer)
    print(f"Saved final checkpoint: {checkpoint_path}")
    if math.isfinite(best_val):
        print(f"Best validation loss: {best_val:.4f}")
    return checkpoint_path

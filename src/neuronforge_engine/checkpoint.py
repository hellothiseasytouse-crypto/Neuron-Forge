"""Safe checkpoint save/load helpers."""
import os
from pathlib import Path

import torch

from .config import ModelConfig
from .model import GPT
from .tokenizer import ByteBPETokenizer


def save_checkpoint(path: str | Path, model: GPT, tokenizer: ByteBPETokenizer,
                    config: dict, step: int, best_val_loss: float,
                    optimizer: torch.optim.Optimizer | None = None) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 2,
        "model_config": config["model"],
        "training_config": config["train"],
        "tokenizer": tokenizer.to_state_dict(),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "step": int(step),
        "best_val_loss": float(best_val_loss),
    }
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, destination)


def read_checkpoint(path: str | Path, device: torch.device | str = "cpu") -> dict:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {file_path}")
    try:
        return torch.load(file_path, map_location=device, weights_only=True)
    except TypeError:  # Compatibility with older supported PyTorch releases.
        return torch.load(file_path, map_location=device)


def load_model(path: str | Path, device: torch.device | str = "cpu"):
    checkpoint = read_checkpoint(path, device=device)
    if checkpoint.get("format_version") != 2:
        raise ValueError("This checkpoint uses the previous Transformer architecture; retrain with the current Neuron Forge model.")
    model_cfg = ModelConfig(**checkpoint["model_config"])
    tokenizer = ByteBPETokenizer.from_state_dict(checkpoint["tokenizer"])
    if tokenizer.vocab_size != model_cfg.vocab_size:
        raise ValueError("Checkpoint tokenizer and model vocabulary sizes do not match")
    model = GPT(model_cfg).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, tokenizer, checkpoint

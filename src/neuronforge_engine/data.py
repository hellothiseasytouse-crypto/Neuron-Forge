"""Text loading and next-token training batches."""
from pathlib import Path

import torch

from .tokenizer import ByteBPETokenizer


def load_text(path: str | Path) -> str:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(
            f"Training text not found: {file_path}. Add a UTF-8 .txt corpus or run scripts/prepare_data.py."
        )
    text = file_path.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(f"Training text is empty: {file_path}")
    return text


def encode_and_split(text: str, tokenizer: ByteBPETokenizer, val_fraction: float = 0.1):
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1")
    ids = tokenizer.encode(text)
    if len(ids) < 4:
        raise ValueError("Corpus is too short; provide more training text")
    split_at = int(len(ids) * (1.0 - val_fraction))
    split_at = min(max(split_at, 2), len(ids) - 2)
    return torch.tensor(ids[:split_at], dtype=torch.long), torch.tensor(ids[split_at:], dtype=torch.long)


def get_batch(data: torch.Tensor, block_size: int, batch_size: int,
              device: torch.device | str = "cpu"):
    if data.ndim != 1:
        raise ValueError("data must be a one-dimensional token tensor")
    if len(data) <= block_size:
        raise ValueError(
            f"Dataset has {len(data)} tokens but block_size is {block_size}; add more text or reduce block_size."
        )
    starts = torch.randint(0, len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in starts])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in starts])
    return x.to(device), y.to(device)

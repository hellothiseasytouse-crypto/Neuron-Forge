"""Turns raw text into training batches."""
import torch


def load_text(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


def split_data(ids: list[int], val_fraction: float = 0.1):
    # TODO: tensor conversion + train/val split
    raise NotImplementedError


def get_batch(data: torch.Tensor, block_size: int, batch_size: int):
    # TODO: sample random windows -> (x, y) where y is x shifted by one
    raise NotImplementedError

"""Decoder-only transformer (GPT-style)."""
import torch.nn as nn
from .config import ModelConfig


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        # TODO: qkv projection, output projection, causal mask
        raise NotImplementedError


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        # TODO: LayerNorm -> attention -> LayerNorm -> MLP (with residuals)
        raise NotImplementedError


class GPT(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        # TODO: token + position embeddings, N blocks, final norm, LM head
        raise NotImplementedError

    def forward(self, idx, targets=None):
        # TODO: return (logits, loss)
        raise NotImplementedError

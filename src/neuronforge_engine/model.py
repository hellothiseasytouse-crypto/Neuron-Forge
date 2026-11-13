"""Decoder-only Transformer with RoPE, RMSNorm, SwiGLU and SDPA attention."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


class RMSNorm(nn.Module):
    """RMS normalization with a learned scale and no unnecessary bias."""
    def __init__(self, width: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = x * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return normalized.to(dtype=x.dtype) * self.weight


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    even = x[..., 0::2]
    odd = x[..., 1::2]
    return torch.stack((-odd, even), dim=-1).flatten(-2)


def _apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return (x * cos) + (_rotate_half(x) * sin)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        if self.head_dim % 2:
            raise ValueError("RoPE requires an even attention head dimension")
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.attn_dropout = cfg.dropout
        self.resid_dropout = nn.Dropout(cfg.dropout)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _rotary_values(self, length: int, device: torch.device, dtype: torch.dtype):
        positions = torch.arange(length, device=device, dtype=self.inv_freq.dtype)
        frequencies = torch.outer(positions, self.inv_freq.to(device=device))
        angles = torch.cat((frequencies, frequencies), dim=-1)
        # Interleave frequencies to match the even/odd channel rotation.
        angles = torch.repeat_interleave(frequencies, 2, dim=-1)
        return angles.cos().to(dtype=dtype)[None, None], angles.sin().to(dtype=dtype)[None, None]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, channels = x.shape
        qkv = self.qkv(x).view(batch, length, 3, self.n_head, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        cos, sin = self._rotary_values(length, x.device, q.dtype)
        q = _apply_rotary(q, cos, sin)
        k = _apply_rotary(k, cos, sin)

        if hasattr(F, "scaled_dot_product_attention"):
            out = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            scores = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
            mask = torch.tril(torch.ones(length, length, device=x.device, dtype=torch.bool))
            scores = scores.masked_fill(~mask[None, None], torch.finfo(scores.dtype).min)
            weights = F.softmax(scores, dim=-1)
            if self.training and self.attn_dropout:
                weights = F.dropout(weights, p=self.attn_dropout)
            out = weights @ v

        out = out.transpose(1, 2).contiguous().view(batch, length, channels)
        return self.resid_dropout(self.proj(out))


class SwiGLU(nn.Module):
    """Gated feed-forward block with a SiLU gate."""
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        hidden = int(8 * cfg.n_embd / 3)
        hidden = ((hidden + 7) // 8) * 8
        self.gate_up = nn.Linear(cfg.n_embd, 2 * hidden, bias=False)
        self.down = nn.Linear(hidden, cfg.n_embd, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, value = self.gate_up(x).chunk(2, dim=-1)
        return self.dropout(self.down(F.silu(gate) * value))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.mlp_norm = RMSNorm(cfg.n_embd)
        self.mlp = SwiGLU(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.mlp(self.mlp_norm(x))
        return x


class GPT(nn.Module):
    """A decoder-only Transformer initialized from scratch (no pretrained weights)."""
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        cfg.validate()
        self.config = cfg
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.final_norm = RMSNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.apply(self._init_weights)
        for name, parameter in self.named_parameters():
            if name.endswith("proj.weight") or name.endswith("mlp.down.weight"):
                nn.init.normal_(parameter, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        if idx.ndim != 2:
            raise ValueError("Input token IDs must have shape [batch, sequence]")
        batch, length = idx.shape
        if length < 1:
            raise ValueError("Input sequence must contain at least one token")
        if length > self.config.block_size:
            raise ValueError(f"Sequence length {length} exceeds block_size {self.config.block_size}")
        x = self.drop(self.token_embedding(idx))
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.final_norm(x))
        loss = None
        if targets is not None:
            if targets.shape != idx.shape:
                raise ValueError("Targets must have the same shape as input token IDs")
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    def num_parameters(self, trainable_only: bool = True) -> int:
        return sum(p.numel() for p in self.parameters() if not trainable_only or p.requires_grad)

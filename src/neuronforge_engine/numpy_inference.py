"""Small, CPU-friendly NumPy inference for an exported Neuron Forge checkpoint.

The Vercel chat function uses this module so it does not need to bundle the
full PyTorch training framework. The model architecture mirrors the trained
RoPE + RMSNorm + SwiGLU Transformer.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class NumpyByteBPE:
    def __init__(self, merges):
        self.merges = [(int(a), int(b)) for a, b in merges]
        self.token_bytes = [bytes([i]) for i in range(256)]
        for left, right in self.merges:
            if left >= len(self.token_bytes) or right >= len(self.token_bytes):
                raise ValueError("The tokenizer merge table is invalid")
            self.token_bytes.append(self.token_bytes[left] + self.token_bytes[right])

    def encode(self, text: str) -> list[int]:
        ids = list(text.encode("utf-8"))
        for index, pair in enumerate(self.merges):
            merged_id = 256 + index
            result = []
            i = 0
            while i < len(ids):
                if i + 1 < len(ids) and (ids[i], ids[i + 1]) == pair:
                    result.append(merged_id)
                    i += 2
                else:
                    result.append(ids[i])
                    i += 1
            ids = result
        return ids

    def decode(self, ids) -> str:
        return b"".join(self.token_bytes[int(i)] for i in ids).decode("utf-8", errors="replace")


class NumpyNeuronForge:
    def __init__(self, checkpoint_dir: str | Path):
        root = Path(checkpoint_dir)
        metadata = json.loads((root / "runtime.json").read_text(encoding="utf-8"))
        if metadata.get("format_version") != 1:
            raise ValueError("Unsupported Neuron Forge NumPy checkpoint format")
        self.config = metadata["model_config"]
        self.vocab_size = int(self.config["vocab_size"])
        self.block_size = int(self.config["block_size"])
        self.n_layer = int(self.config["n_layer"])
        self.n_head = int(self.config["n_head"])
        self.n_embd = int(self.config["n_embd"])
        self.head_dim = self.n_embd // self.n_head
        if self.n_embd % self.n_head or self.head_dim % 2:
            raise ValueError("Checkpoint head dimensions are incompatible with RoPE")
        self.tokenizer = NumpyByteBPE(metadata["tokenizer"]["merges"])
        self.weights = np.load(root / "runtime.npz", allow_pickle=False)
        self.embedding = self.weights["token_embedding_weight"].astype(np.float32, copy=False)
        if self.embedding.shape != (self.vocab_size, self.n_embd):
            raise ValueError("Checkpoint token embedding dimensions do not match its config")
        self.inv_freq = 1.0 / (10000 ** (np.arange(0, self.head_dim, 2, dtype=np.float32) / self.head_dim))
        self.checkpoint_step = int(metadata.get("checkpoint_step", 0))
        self.rng = np.random.default_rng()

    @staticmethod
    def _linear(x: np.ndarray, weight: np.ndarray) -> np.ndarray:
        return x @ weight.T

    @staticmethod
    def _rms_norm(x: np.ndarray, weight: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        scale = 1.0 / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
        return (x * scale) * weight

    @staticmethod
    def _rotate_half(x: np.ndarray) -> np.ndarray:
        out = np.empty_like(x)
        out[..., 0::2] = -x[..., 1::2]
        out[..., 1::2] = x[..., 0::2]
        return out

    def logits(self, token_ids) -> np.ndarray:
        ids = np.asarray(token_ids, dtype=np.int64)
        if ids.ndim != 1 or ids.size < 1:
            raise ValueError("Expected a non-empty one-dimensional token sequence")
        ids = ids[-self.block_size:]
        x = self.embedding[ids].copy()
        seq_len = x.shape[0]
        positions = np.arange(seq_len, dtype=np.float32)
        angles = positions[:, None] * self.inv_freq[None, :]
        angles = np.repeat(angles, 2, axis=-1)
        cos = np.cos(angles).astype(np.float32)[None, :, :]
        sin = np.sin(angles).astype(np.float32)[None, :, :]
        causal = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)

        for layer in range(self.n_layer):
            prefix = f"blocks_{layer}_"
            norm = self._rms_norm(x, self.weights[prefix + "attn_norm_weight"])
            qkv = self._linear(norm, self.weights[prefix + "attn_qkv_weight"])
            qkv = qkv.reshape(seq_len, 3, self.n_head, self.head_dim)
            q, k, v = (qkv[:, i].transpose(1, 0, 2) for i in range(3))
            q = q * cos + self._rotate_half(q) * sin
            k = k * cos + self._rotate_half(k) * sin
            scores = (q @ k.transpose(0, 2, 1)) * (1.0 / np.sqrt(self.head_dim))
            scores = np.where(causal[None, :, :], -1e30, scores)
            scores -= np.max(scores, axis=-1, keepdims=True)
            attention = np.exp(scores)
            attention /= np.sum(attention, axis=-1, keepdims=True)
            attended = attention @ v
            attended = attended.transpose(1, 0, 2).reshape(seq_len, self.n_embd)
            x = x + self._linear(attended, self.weights[prefix + "attn_proj_weight"])

            norm = self._rms_norm(x, self.weights[prefix + "mlp_norm_weight"])
            gate_value = self._linear(norm, self.weights[prefix + "mlp_gate_up_weight"])
            gate, value = np.split(gate_value, 2, axis=-1)
            silu = gate / (1.0 + np.exp(-np.clip(gate, -80.0, 80.0)))
            x = x + self._linear(silu * value, self.weights[prefix + "mlp_down_weight"])

        x = self._rms_norm(x, self.weights["final_norm_weight"])
        return x @ self.embedding.T

    def generate(self, prompt: str, max_new_tokens: int = 64, temperature: float = 0.85,
                 top_k: int = 32, top_p: float = 0.92) -> str:
        if not 1 <= max_new_tokens <= 96:
            raise ValueError("max_new_tokens must be between 1 and 96")
        if not 0.05 <= temperature <= 2.0:
            raise ValueError("temperature must be between 0.05 and 2.0")
        ids = self.tokenizer.encode(prompt) or [10]
        for _ in range(max_new_tokens):
            scores = self.logits(ids)[-1].astype(np.float64)
            scores /= temperature
            k = max(1, min(int(top_k), scores.size))
            keep = np.argpartition(scores, -k)[-k:]
            filtered = np.full_like(scores, -np.inf)
            filtered[keep] = scores[keep]
            order = np.argsort(filtered)[::-1]
            ordered = filtered[order]
            probs = np.exp(ordered - np.max(ordered))
            probs /= probs.sum()
            cumulative = np.cumsum(probs)
            cutoff = int(np.searchsorted(cumulative, top_p, side="left")) + 1
            cutoff = max(1, min(cutoff, len(order)))
            allowed = order[:cutoff]
            allowed_probs = probs[:cutoff]
            allowed_probs = allowed_probs / allowed_probs.sum()
            ids.append(int(self.rng.choice(allowed, p=allowed_probs)))
            if len(ids) > self.block_size:
                ids = ids[-self.block_size:]
        return self.tokenizer.decode(ids)

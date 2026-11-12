"""A small, deterministic byte-level BPE tokenizer implemented in Python.

The first 256 token IDs represent raw bytes. Every learned token is a merge of
an earlier pair, so any UTF-8 input can always be encoded without an UNK token.
"""
from collections import Counter
from dataclasses import dataclass
from typing import Iterable


@dataclass
class ByteBPETokenizer:
    merges: list[tuple[int, int]]
    token_bytes: list[bytes]

    def __init__(self, merges: Iterable[tuple[int, int]] | None = None):
        self.merges = []
        self.token_bytes = [bytes([i]) for i in range(256)]
        for pair in merges or []:
            self._append_merge((int(pair[0]), int(pair[1])))

    def _append_merge(self, pair: tuple[int, int]) -> None:
        left, right = pair
        if left < 0 or right < 0 or left >= len(self.token_bytes) or right >= len(self.token_bytes):
            raise ValueError(f"Invalid BPE merge references: {pair}")
        self.merges.append(pair)
        self.token_bytes.append(self.token_bytes[left] + self.token_bytes[right])

    @classmethod
    def train(cls, text: str, vocab_size: int = 512, min_frequency: int = 2,
              max_train_chars: int = 2_000_000) -> "ByteBPETokenizer":
        if vocab_size < 256:
            raise ValueError("vocab_size must be at least 256")
        if min_frequency < 2:
            raise ValueError("min_frequency must be at least 2")
        if max_train_chars > 0:
            text = text[:max_train_chars]
        tokens = list(text.encode("utf-8"))
        learned: list[tuple[int, int]] = []
        next_id = 256
        for _ in range(vocab_size - 256):
            if len(tokens) < 2:
                break
            counts = Counter(zip(tokens, tokens[1:]))
            if not counts:
                break
            pair, count = max(counts.items(), key=lambda item: (item[1], item[0]))
            if count < min_frequency:
                break
            merged: list[int] = []
            i = 0
            while i < len(tokens):
                if i + 1 < len(tokens) and (tokens[i], tokens[i + 1]) == pair:
                    merged.append(next_id)
                    i += 2
                else:
                    merged.append(tokens[i])
                    i += 1
            tokens = merged
            learned.append(pair)
            next_id += 1
        return cls(learned)

    @classmethod
    def from_text(cls, text: str, vocab_size: int = 512) -> "ByteBPETokenizer":
        """Compatibility helper for callers that prefer a from_text constructor."""
        return cls.train(text, vocab_size=vocab_size)

    @property
    def vocab_size(self) -> int:
        return len(self.token_bytes)

    def encode(self, text: str) -> list[int]:
        ids = list(text.encode("utf-8"))
        for merge_index, pair in enumerate(self.merges):
            merged_id = 256 + merge_index
            out: list[int] = []
            i = 0
            while i < len(ids):
                if i + 1 < len(ids) and (ids[i], ids[i + 1]) == pair:
                    out.append(merged_id)
                    i += 2
                else:
                    out.append(ids[i])
                    i += 1
            ids = out
        return ids

    def decode(self, ids: Iterable[int]) -> str:
        chunks: list[bytes] = []
        for token_id in ids:
            if token_id < 0 or token_id >= self.vocab_size:
                raise ValueError(f"Token ID out of range: {token_id}")
            chunks.append(self.token_bytes[token_id])
        return b"".join(chunks).decode("utf-8", errors="replace")

    def to_state_dict(self) -> dict:
        return {"merges": [list(pair) for pair in self.merges]}

    @classmethod
    def from_state_dict(cls, state: dict) -> "ByteBPETokenizer":
        if not isinstance(state, dict) or not isinstance(state.get("merges", []), list):
            raise ValueError("Invalid tokenizer checkpoint")
        return cls([tuple(pair) for pair in state.get("merges", [])])

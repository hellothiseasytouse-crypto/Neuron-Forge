"""Fast byte-level BPE tokenizer for Neuron Forge.

New checkpoints store a Hugging Face Tokenizers JSON payload. The legacy
pure-Python merge format remains readable so older experimental checkpoints
continue to work.
"""
from collections import Counter
from typing import Iterable


class ByteBPETokenizer:
    def __init__(self, merges: Iterable[tuple[int, int]] | None = None,
                 tokenizer_json: str | None = None):
        self._fast = None
        self.merges = []
        self.token_bytes = [bytes([i]) for i in range(256)]

        if tokenizer_json is not None:
            try:
                from tokenizers import Tokenizer
            except ImportError as exc:
                raise RuntimeError(
                    "This checkpoint uses the fast BPE tokenizer. Install the project dependencies."
                ) from exc
            self._fast = Tokenizer.from_str(tokenizer_json)
            return

        for pair in merges or []:
            left, right = int(pair[0]), int(pair[1])
            if left < 0 or right < 0 or left >= len(self.token_bytes) or right >= len(self.token_bytes):
                raise ValueError(f"Invalid BPE merge references: {(left, right)}")
            self.merges.append((left, right))
            self.token_bytes.append(self.token_bytes[left] + self.token_bytes[right])

    @classmethod
    def train(cls, text: str, vocab_size: int = 512, min_frequency: int = 2,
              max_train_chars: int = 2_000_000) -> "ByteBPETokenizer":
        if vocab_size < 257:
            raise ValueError("vocab_size must be at least 257 for byte-level BPE")
        if min_frequency < 2:
            raise ValueError("min_frequency must be at least 2")
        sample = text[:max_train_chars] if max_train_chars > 0 else text
        if not sample.strip():
            raise ValueError("Cannot train a tokenizer on empty text")

        try:
            from tokenizers import Tokenizer
            from tokenizers.decoders import ByteLevel
            from tokenizers.models import BPE
            from tokenizers.pre_tokenizers import ByteLevel as ByteLevelPreTokenizer
            from tokenizers.trainers import BpeTrainer
        except ImportError as exc:
            raise RuntimeError(
                "Fast tokenizer support is missing. Install with: pip install tokenizers"
            ) from exc

        tokenizer = Tokenizer(BPE(unk_token="<unk>"))
        tokenizer.pre_tokenizer = ByteLevelPreTokenizer(add_prefix_space=False)
        tokenizer.decoder = ByteLevel()
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=["<unk>"],
            initial_alphabet=ByteLevelPreTokenizer.alphabet(),
            max_token_length=64,
            show_progress=True,
        )

        chunk_size = 250_000
        chunks = (sample[i:i + chunk_size] for i in range(0, len(sample), chunk_size))
        tokenizer.train_from_iterator(
            chunks,
            trainer=trainer,
            length=(len(sample) + chunk_size - 1) // chunk_size,
        )
        result = cls(tokenizer_json=tokenizer.to_str())
        if result.vocab_size != vocab_size:
            raise ValueError(
                f"Tokenizer learned {result.vocab_size:,} tokens; expected {vocab_size:,}. "
                "Provide more diverse training text."
            )
        return result

    @classmethod
    def from_text(cls, text: str, vocab_size: int = 512) -> "ByteBPETokenizer":
        return cls.train(text, vocab_size=vocab_size)

    @property
    def vocab_size(self) -> int:
        if self._fast is not None:
            return self._fast.get_vocab_size(with_added_tokens=True)
        return len(self.token_bytes)

    def encode(self, text: str) -> list[int]:
        if self._fast is not None:
            return self._fast.encode(text).ids
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
        values = [int(token_id) for token_id in ids]
        if self._fast is not None:
            return self._fast.decode(values, skip_special_tokens=False)
        chunks: list[bytes] = []
        for token_id in values:
            if token_id < 0 or token_id >= self.vocab_size:
                raise ValueError(f"Token ID out of range: {token_id}")
            chunks.append(self.token_bytes[token_id])
        return b"".join(chunks).decode("utf-8", errors="replace")

    def to_state_dict(self) -> dict:
        if self._fast is not None:
            return {"tokenizer_json": self._fast.to_str()}
        return {"merges": [list(pair) for pair in self.merges]}

    @classmethod
    def from_state_dict(cls, state: dict) -> "ByteBPETokenizer":
        if not isinstance(state, dict):
            raise ValueError("Invalid tokenizer checkpoint")
        if isinstance(state.get("tokenizer_json"), str):
            return cls(tokenizer_json=state["tokenizer_json"])
        merges = state.get("merges", [])
        if not isinstance(merges, list):
            raise ValueError("Invalid legacy tokenizer checkpoint")
        return cls([tuple(pair) for pair in merges])

"""Validated configuration for training and running Neuron Forge."""
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    vocab_size: int = 512
    block_size: int = 256
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.1

    def parameter_count(self) -> int:
        """Return the exact trainable parameter count for the tied-embedding GPT."""
        hidden = ((8 * self.n_embd // 3 + 7) // 8) * 8
        per_block = (
            3 * self.n_embd * self.n_embd
            + self.n_embd * self.n_embd
            + 2 * self.n_embd * hidden
            + self.n_embd * hidden
            + 2 * self.n_embd
        )
        return self.vocab_size * self.n_embd + self.n_layer * per_block + self.n_embd

    def validate(self) -> None:
        if self.vocab_size < 256:
            raise ValueError("vocab_size must be at least 256 for byte-level tokenization")
        if self.block_size < 8:
            raise ValueError("block_size must be at least 8")
        if self.n_layer < 1 or self.n_head < 1 or self.n_embd < 8:
            raise ValueError("n_layer, n_head and n_embd must be positive")
        if self.n_embd % self.n_head:
            raise ValueError("n_embd must be divisible by n_head")
        if (self.n_embd // self.n_head) % 2:
            raise ValueError("each attention head must have an even width for rotary embeddings")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")


@dataclass
class TokenizerConfig:
    vocab_size: int = 512
    min_frequency: int = 2
    max_train_chars: int = 2_000_000

    def validate(self) -> None:
        if self.vocab_size < 256:
            raise ValueError("tokenizer.vocab_size must be at least 256")
        if self.min_frequency < 2:
            raise ValueError("tokenizer.min_frequency must be at least 2")
        if self.max_train_chars < 1:
            raise ValueError("tokenizer.max_train_chars must be positive")


@dataclass
class TrainConfig:
    data_path: str = "data/processed/shakespeare_corpus.txt"
    batch_size: int = 16
    max_steps: int = 5_000
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    warmup_steps: int = 100
    eval_interval: int = 250
    eval_batches: int = 20
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    gradient_accumulation_steps: int = 1
    gradient_checkpointing: bool = False
    checkpoint_path: str = "checkpoints/latest.pt"
    seed: int = 1337
    device: str = "auto"

    def validate(self) -> None:
        if self.batch_size < 1 or self.max_steps < 1:
            raise ValueError("batch_size and max_steps must be positive")
        if self.learning_rate <= 0 or self.min_learning_rate < 0:
            raise ValueError("learning rates must be positive (minimum may be zero)")
        if self.warmup_steps < 0 or self.eval_interval < 1 or self.eval_batches < 1:
            raise ValueError("warmup_steps must be non-negative and eval settings positive")
        if self.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be positive")
        if self.grad_clip < 0 or self.weight_decay < 0:
            raise ValueError("grad_clip and weight_decay must be non-negative")
        if self.device not in {"auto", "cpu", "cuda", "mps"}:
            raise ValueError("device must be one of: auto, cpu, cuda, mps")


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def validate(self) -> None:
        self.model.validate()
        self.tokenizer.validate()
        self.train.validate()
        if self.model.vocab_size != self.tokenizer.vocab_size:
            self.model.vocab_size = self.tokenizer.vocab_size

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config(path: str | Path) -> Config:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration not found: {config_path}")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("Configuration root must be a YAML mapping")
    cfg = Config(
        model=ModelConfig(**raw.get("model", {})),
        tokenizer=TokenizerConfig(**raw.get("tokenizer", {})),
        train=TrainConfig(**raw.get("train", {})),
    )
    cfg.validate()
    return cfg

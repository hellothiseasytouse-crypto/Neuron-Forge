from dataclasses import dataclass, field
import yaml


@dataclass
class ModelConfig:
    vocab_size: int = 0  # set from the tokenizer at train time
    block_size: int = 128
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.1


@dataclass
class TrainConfig:
    data_path: str = "data/raw/input.txt"
    batch_size: int = 32
    max_steps: int = 5000
    learning_rate: float = 3e-4
    eval_interval: int = 250
    checkpoint_path: str = "checkpoints/latest.pt"
    seed: int = 1337


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


def load_config(path: str) -> Config:
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    return Config(
        model=ModelConfig(**raw.get("model", {})),
        train=TrainConfig(**raw.get("train", {})),
    )

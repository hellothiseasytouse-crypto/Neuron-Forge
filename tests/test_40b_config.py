from pathlib import Path

from neuronforge_engine.config import load_config


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_40b_config_has_exact_parameter_count_and_training_target():
    config = load_config(REPO_ROOT / "configs" / "40b.yaml")
    assert config.model.parameter_count() == 40_000_905_216
    assert config.model.n_layer == 49
    assert config.model.n_head == 64
    assert config.model.n_embd == 8192
    assert config.model.block_size == 2048
    assert config.tokenizer.vocab_size == 65536
    assert config.train.target_tokens == 800_000_000_000
    assert config.train.gradient_checkpointing is True

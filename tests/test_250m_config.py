import json
from pathlib import Path

import torch

from neuronforge_engine.config import ModelConfig, load_config
from neuronforge_engine.model import GPT


ROOT = Path(__file__).resolve().parents[1]


def test_250m_architecture_has_exact_target_parameter_count():
    config = load_config(ROOT / "configs" / "250m.yaml")
    assert config.model.parameter_count() == 248_144_832
    assert config.model.n_layer == 29
    assert config.model.n_head == 16
    assert config.model.n_embd == 832
    assert config.model.block_size == 1024
    assert config.model.vocab_size == 8192
    assert config.train.device == "cuda"
    assert config.train.gradient_checkpointing is True


def test_architecture_manifest_matches_training_config():
    manifest = json.loads((ROOT / "configs" / "250m-architecture.json").read_text())
    config = load_config(ROOT / "configs" / "250m.yaml")
    assert manifest["parameters"] == config.model.parameter_count()
    assert manifest["model"]["n_layer"] == config.model.n_layer
    assert manifest["model"]["vocab_size"] == config.model.vocab_size
    assert manifest["status"] == "architecture_configured_training_not_started"


def test_parameter_formula_matches_real_small_model():
    config = ModelConfig(
        vocab_size=256,
        block_size=16,
        n_layer=2,
        n_head=2,
        n_embd=16,
        dropout=0.0,
    )
    model = GPT(config)
    assert config.parameter_count() == model.num_parameters()


def test_activation_checkpointing_supports_backward_pass():
    config = ModelConfig(
        vocab_size=256,
        block_size=16,
        n_layer=2,
        n_head=2,
        n_embd=16,
        dropout=0.0,
    )
    model = GPT(config)
    model.train()
    model.gradient_checkpointing = True
    tokens = torch.randint(0, config.vocab_size, (2, 12))
    _, loss = model(tokens, tokens)
    loss.backward()
    assert loss is not None
    assert model.token_embedding.weight.grad is not None

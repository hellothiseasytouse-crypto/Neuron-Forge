import json
from pathlib import Path

from neuronforge_engine.config import load_config


ROOT = Path(__file__).resolve().parents[1]


def test_250m_architecture_has_exact_target_parameter_count():
    config = load_config(ROOT / "configs" / "250m.yaml")
    assert config.model.parameter_count() == 248_144_832
    assert config.model.n_layer == 29
    assert config.model.n_head == 16
    assert config.model.n_embd == 832
    assert config.model.block_size == 1024
    assert config.train.device == "cuda"
    assert config.train.gradient_checkpointing is True


def test_architecture_manifest_matches_training_config():
    manifest = json.loads((ROOT / "configs" / "250m-architecture.json").read_text())
    config = load_config(ROOT / "configs" / "250m.yaml")
    assert manifest["parameters"] == config.model.parameter_count()
    assert manifest["model"]["n_layer"] == config.model.n_layer
    assert manifest["training"]["status"] == "not_started"

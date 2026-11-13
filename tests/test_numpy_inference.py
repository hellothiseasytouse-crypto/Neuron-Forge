import json
from pathlib import Path

import numpy as np

from neuronforge_engine.numpy_inference import NumpyByteBPE, NumpyNeuronForge


def test_byte_bpe_round_trip_with_a_merge():
    tokenizer = NumpyByteBPE([[104, 105]])
    assert tokenizer.encode("hi") == [256]
    assert tokenizer.decode([256]) == "hi"


def test_numpy_runtime_loads_runs_forward_and_generates(tmp_path: Path):
    config = {
        "vocab_size": 256,
        "block_size": 16,
        "n_layer": 1,
        "n_head": 1,
        "n_embd": 8,
        "dropout": 0.0,
    }
    metadata = {
        "format_version": 1,
        "checkpoint_step": 7,
        "model_config": config,
        "tokenizer": {"merges": []},
    }
    (tmp_path / "runtime.json").write_text(json.dumps(metadata), encoding="utf-8")
    rng = np.random.default_rng(4)
    np.savez_compressed(
        tmp_path / "runtime.npz",
        token_embedding_weight=rng.normal(size=(256, 8)).astype(np.float32),
        blocks_0_attn_norm_weight=np.ones(8, dtype=np.float32),
        blocks_0_attn_qkv_weight=rng.normal(size=(24, 8)).astype(np.float32) * 0.02,
        blocks_0_attn_proj_weight=rng.normal(size=(8, 8)).astype(np.float32) * 0.02,
        blocks_0_mlp_norm_weight=np.ones(8, dtype=np.float32),
        blocks_0_mlp_gate_up_weight=rng.normal(size=(48, 8)).astype(np.float32) * 0.02,
        blocks_0_mlp_down_weight=rng.normal(size=(8, 24)).astype(np.float32) * 0.02,
        final_norm_weight=np.ones(8, dtype=np.float32),
    )

    model = NumpyNeuronForge(tmp_path)
    logits = model.logits(model.tokenizer.encode("hi"))
    assert logits.shape == (2, 256)
    assert np.isfinite(logits).all()
    assert model.checkpoint_step == 7
    assert model.generate("hi", max_new_tokens=3).startswith("hi")

"""Export a Neuron Forge PyTorch checkpoint into compact NumPy inference files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if checkpoint.get("format_version") != 2:
        raise SystemExit("Unsupported checkpoint format")

    weights = {}
    for key, tensor in checkpoint["model_state"].items():
        safe_key = key.replace(".", "_")
        weights[safe_key] = tensor.detach().cpu().float().numpy()
    np.savez_compressed(output / "runtime.npz", **weights)
    metadata = {
        "format_version": 1,
        "checkpoint_step": int(checkpoint.get("step", 0)),
        "best_validation_loss": float(checkpoint.get("best_val_loss", 0.0)),
        "model_config": checkpoint["model_config"],
        "tokenizer": checkpoint["tokenizer"],
    }
    (output / "runtime.json").write_text(json.dumps(metadata, separators=(",", ":")), encoding="utf-8")
    print(f"Exported step {metadata['checkpoint_step']} to {output}")
    print(f"Runtime archive: {(output / 'runtime.npz').stat().st_size:,} bytes")


if __name__ == "__main__":
    main()

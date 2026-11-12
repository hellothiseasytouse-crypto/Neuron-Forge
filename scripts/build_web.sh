#!/usr/bin/env bash
set -euo pipefail

# The hosted demo uses only the Python standard library. PyTorch remains an
# optional dependency for full GPT training outside the serverless web build.
python scripts/prepare_data.py
python scripts/train_web_model.py
test -s public/model.json
echo "Neuron Forge web model built successfully."

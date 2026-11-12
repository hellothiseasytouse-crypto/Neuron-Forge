#!/usr/bin/env bash
set -euo pipefail

# Vercel resolves the project's Python dependencies from pyproject.toml and
# requirements.txt before running this build command. Do not run pip here:
# its build Python environment is managed by uv (PEP 668).
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"

python scripts/prepare_data.py
python -c "import torch; torch.set_num_threads(1); from neuronforge_engine.cli import main; main(['train', '--config', 'configs/quick.yaml'])"

test -s checkpoints/latest.pt
echo "Neuron Forge demo checkpoint built successfully."

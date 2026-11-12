#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"

python -m pip install --no-cache-dir -r requirements.txt
python -m pip install --no-deps -e .

python scripts/prepare_data.py
python -c "import torch; torch.set_num_threads(1); from neuronforge_engine.cli import main; main(['train', '--config', 'configs/quick.yaml'])"

test -s checkpoints/latest.pt
echo "Neuron Forge demo checkpoint built successfully."

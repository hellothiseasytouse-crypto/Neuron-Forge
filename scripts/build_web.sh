#!/usr/bin/env bash
set -euo pipefail

ROOT="$(pwd)"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

python "$ROOT/scripts/prepare_data.py"

# Import torch outside the source checkout first. This avoids a local source
# directory shadowing the installed PyTorch package in build environments.
(cd /tmp && PYTHONPATH="$ROOT/src" python -c "import torch; print('PyTorch', torch.__version__); torch.set_num_threads(1)")

python -c "import torch; torch.set_num_threads(1); from neuronforge_engine.cli import main; main(['train', '--config', '$ROOT/configs/quick.yaml'])"

test -s "$ROOT/checkpoints/latest.pt"
echo "Neuron Forge demo checkpoint built successfully."

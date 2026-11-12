#!/usr/bin/env bash
set -euo pipefail

ROOT="$(pwd)"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONPATH="$ROOT/.vercel_python_packages:$ROOT/src"

python "$ROOT/scripts/prepare_data.py"

# Vercel installs Python wheels into .vercel_python_packages. Launching outside
# the checkout prevents repository paths from shadowing compiled packages.
sed \
  -e "s|data/raw/input.txt|$ROOT/data/raw/input.txt|g" \
  -e "s|checkpoints/latest.pt|$ROOT/checkpoints/latest.pt|g" \
  "$ROOT/configs/quick.yaml" > /tmp/neuronforge-quick.yaml

cd /tmp
python -c "import torch; print('PyTorch', torch.__version__); torch.set_num_threads(1)"
python -c "import torch; torch.set_num_threads(1); from neuronforge_engine.cli import main; main(['train', '--config', '/tmp/neuronforge-quick.yaml'])"

test -s "$ROOT/checkpoints/latest.pt"
echo "Neuron Forge demo checkpoint built successfully."

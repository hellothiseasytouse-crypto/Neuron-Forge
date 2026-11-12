#!/usr/bin/env bash
set -euo pipefail

# Build a searchable local index from the source text files committed in this repository.
python scripts/prepare_data.py
python scripts/build_corpus_index.py
test -s public/shakespeare_index.json
echo "Neuron Forge Shakespeare corpus index built successfully."

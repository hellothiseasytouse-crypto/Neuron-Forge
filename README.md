# Neuron Forge

**An experimental language model built from random initialization.** Neuron Forge implements its own byte-level BPE tokenizer, decoder-only Transformer, training loop, checkpoint format, generation sampler, and local interfaces. It does not load pretrained model weights or call a hosted AI API.

> **Status:** working educational training engine. A functioning codebase is not the same thing as a trained, capable general-purpose AI. Model quality depends on corpus quality, model size, training time, and available compute.

## What is included

- Byte-level BPE tokenizer with a 256-byte base vocabulary and learned merges.
- GPT-style causal Transformer with multi-head self-attention, MLP blocks, residual connections, and tied embeddings.
- Training from random initialization using AdamW, warmup/cosine learning-rate scheduling, gradient accumulation, clipping, validation loss, and checkpoints.
- Generation with temperature, top-k, top-p, and repetition penalty.
- CLI for training, generation, terminal chat, and local serving.
- Optional FastAPI endpoint and browser chat page.
- Tests and GitHub Actions CI.

## Quick start

Requirements: Python 3.10+ and a compatible PyTorch installation.

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"

# Create a tiny example corpus (for pipeline testing only)
python scripts/prepare_data.py

# Train a small model from scratch
neuronforge train --config configs/quick.yaml

# Generate text
neuronforge generate --checkpoint checkpoints/latest.pt --prompt "Hello"
```

## Train on your own dataset

Put UTF-8 `.txt` files in a local folder, then combine them:

```bash
python scripts/prepare_corpus.py --input ./my-text-files --output data/processed/corpus.txt
```

Set `train.data_path` in `configs/base.yaml` to `data/processed/corpus.txt`, review the model/training sizes for your hardware, and run:

```bash
neuronforge train --config configs/base.yaml
```

Resume a run with `neuronforge train --config configs/base.yaml --resume`. Checkpoints are written to `checkpoints/`, which is ignored by Git.

## Local browser chat

```bash
pip install -e ".[web]"
neuronforge serve --checkpoint checkpoints/latest.pt
```

Open `http://127.0.0.1:8000`. The server binds to localhost by default. Do not expose it to the public internet without authentication, rate limiting, TLS, and a security review.

## Repository structure

```text
src/neuronforge/             # original starter files supplied in the ZIP (kept unchanged)
src/neuronforge_engine/      # complete training and inference implementation
  config.py                  # validated model and training configuration
  tokenizer.py               # byte-level BPE
  data.py                    # corpus encoding and batch sampling
  model.py                   # causal GPT-style Transformer
  training.py                # optimization, evaluation and checkpointing
  generation.py              # sampling and text generation
  checkpoint.py              # checkpoint serialization and restore
  cli.py                     # train / generate / chat / serve commands
  api.py                     # optional local FastAPI API
  static/index.html          # local browser interface
configs/                     # quick and base training profiles
scripts/                     # corpus preparation tools
 tests/                      # unit tests
 docs/                       # architecture, training, data and roadmap notes
```

## Important limitations

The included sample corpus is repetitive and exists only to smoke-test the pipeline. It will not produce an intelligent assistant. A capable foundation model requires large and diverse licensed datasets, substantial GPU compute, long training runs, evaluation, and safety testing. This project is a real from-scratch training implementation, not a claim that those requirements have already been met.

See [docs/TRAINING_GUIDE.md](docs/TRAINING_GUIDE.md), [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md), and [docs/DATA_AND_SAFETY.md](docs/DATA_AND_SAFETY.md).

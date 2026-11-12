# Neuron Forge

**An independent language-model research project.** Neuron Forge contains a from-scratch decoder-only Transformer, byte-level BPE tokenizer, training loop, checkpoint tools, and local inference interfaces. The hosted website currently uses a separate Shakespeare corpus engine that searches the text files stored in this repository.

## Hosted website

Open **[Neuron Forge](https://neuron-forge-cyber-0ab0.vercel.app)**.

The website does not call an external AI API. Its search index is built from the local files under `data/shakespeare/`. The current collection contains 37 plays, all 154 sonnets, and the associated poetry files present in the selected text edition. It returns relevant passages from those files instead of sampling random characters from an undertrained network.

The website's corpus engine is a text retrieval system, not a fully trained general-purpose chatbot. It is deliberately described that way in the interface.

## What is included

- Byte-level BPE tokenizer with a 256-byte base vocabulary and learned merges.
- GPT-style causal Transformer with multi-head self-attention, MLP blocks, residual connections, and tied embeddings.
- Training from random initialization using AdamW, warmup/cosine learning-rate scheduling, gradient accumulation, clipping, validation loss, and checkpoints.
- Generation with temperature, top-k, top-p, and repetition penalty.
- CLI for training, generation, terminal chat, and local serving.
- An offline Shakespeare text collection, organized by play and poetry.
- A local corpus preparation and passage-indexing pipeline for the hosted site.
- Tests and GitHub Actions CI.

## Quick start

Requirements: Python 3.10+ and a compatible PyTorch installation.

```bash
python -m venv .venv
# Windows PowerShell: .venv\\Scripts\\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev,training]"

# Assemble the bundled works into a training corpus
python scripts/prepare_data.py

# Train a small Transformer from random initialization
neuronforge train --config configs/quick.yaml

# Generate text
neuronforge generate --checkpoint checkpoints/latest.pt --prompt "Hello"
```

## Train on the bundled Shakespeare corpus

The source files are stored separately under `data/shakespeare/plays/` and `data/shakespeare/poetry/`. Prepare the combined corpus with:

```bash
python scripts/prepare_data.py
```

The combined file is written to `data/processed/shakespeare_corpus.txt`. Review the model and training sizes in `configs/` for your hardware before starting a run.

To rebuild the website's local search index:

```bash
python scripts/build_corpus_index.py
```

## Local browser chat

```bash
pip install -e ".[web]"
neuronforge serve --checkpoint checkpoints/latest.pt
```

Open `http://127.0.0.1:8000`. The server binds to localhost by default. Do not expose it to the public internet without authentication, rate limiting, TLS, and a security review.

## Repository structure

```text
src/neuronforge/                  # original starter files supplied in the ZIP (kept unchanged)
src/neuronforge_engine/           # GPT training and local corpus engine
  config.py                       # validated model and training configuration
  tokenizer.py                    # byte-level BPE
  data.py                         # corpus encoding and batch sampling
  model.py                        # causal GPT-style Transformer
  training.py                     # optimization, evaluation and checkpointing
  generation.py                   # sampling and text generation
  checkpoint.py                   # checkpoint serialization and restore
  web_runtime.py                  # local Shakespeare passage search
  cli.py                           # train / generate / chat / serve commands
api/                               # hosted web function
scripts/                           # corpus preparation and local index generation
data/shakespeare/plays/            # individual play texts
data/shakespeare/poetry/           # sonnets and other poetry
configs/                           # quick and base training profiles
tests/                             # unit tests
docs/                              # architecture, training, data and roadmap notes
public/                            # website UI, icon, styles, and browser script
```

## Limitations

The hosted passage engine returns material from the bundled collection; it does not create an original response to every possible question. The full Transformer code is a genuine training implementation, but a capable general-purpose model still requires much more training data, compute, evaluation, and safety work than this repository currently includes.

The collection is an edition of surviving and attributed works. Lost works cannot be reconstructed, and some collaborative or disputed attributions cannot be settled by a text corpus.

See [docs/TRAINING_GUIDE.md](docs/TRAINING_GUIDE.md), [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md), and [docs/DATA_AND_SAFETY.md](docs/DATA_AND_SAFETY.md).

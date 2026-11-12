# Neuron Forge

**An offline AI research project organized around the original Project Dying Ember FAI source document.** Neuron Forge contains a from-scratch decoder-only Transformer, byte-level BPE tokenizer, training loop, checkpoint tools, a local conversation prototype, and a local Shakespeare text library.

## Hosted website

Open **[Neuron Forge](https://neuron-forge-cyber-0ab0.vercel.app)**.

The chat endpoint does not call Vercel AI Gateway or any external language-model API. It uses the Python offline conversation runtime in `src/neuronforge_engine/offline_chat.py`. The current runtime supports basic conversation, in-session recall, safe arithmetic, source-grounded answers about the Dying Ember document, and local Shakespeare passage lookup.

**Important distinction:** this offline runtime is a working conversational prototype, not a trained general-purpose neural language model. It does not have broad world knowledge or reliable open-ended reasoning. The from-scratch Transformer implementation is separate and still needs a suitable training corpus, sufficient compute, training, and evaluation before it can replace the prototype.

## FAI source document

The exact recovered source document is preserved at:

- `docs/FAI_SOURCE_DYING_EMBER.txt`

That file is stored verbatim. Do not rewrite, paraphrase, normalize, or silently change its language. The four primary systems are the source-defined:

- `PERSONALITY-AI`
- `FEELINGS-AI`
- `RULE-AI`
- `TRAITS-AI`

Together, the source identifies the final reconstruction as `FAI-AI`. The runtime treats the source's evidence-versus-assumption warning as a hard boundary. The document describes the mission and four systems; it is not itself the underlying archive of subject conversations. The runtime must not invent missing messages, memories, emotional states, or traits.

## What is included

- Offline conversation endpoint with no remote model/API dependency.
- FAI source document preserved verbatim.
- Four explicit FAI-system boundaries in the offline runtime.
- In-session conversation history handling.
- Safe, constrained arithmetic parsing (no `eval`).
- Local Shakespeare corpus retrieval as optional reference material.
- Byte-level BPE tokenizer with a 256-byte base vocabulary and learned merges.
- GPT-style causal Transformer with multi-head self-attention, MLP blocks, residual connections, and tied embeddings.
- Training from random initialization using AdamW, warmup/cosine learning-rate scheduling, gradient accumulation, clipping, validation loss, and checkpoints.
- CLI for training, generation, terminal chat, and local serving.
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

The quick profile is only a pipeline smoke test. It will not produce a capable conversational model.

## Run the local browser interface

```bash
pip install -e ".[web]"
neuronforge serve --checkpoint checkpoints/latest.pt
```

Open `http://127.0.0.1:8000`. The server binds to localhost by default. Do not expose it to the public internet without authentication, rate limiting, TLS, and a security review.

## Repository structure

```text
src/neuronforge_engine/
  offline_chat.py                 # local dialogue routing and FAI source boundaries
  config.py                       # model and training configuration
  tokenizer.py                    # byte-level BPE
  data.py                         # corpus encoding and batch sampling
  model.py                         # causal GPT-style Transformer
  training.py                     # optimization, evaluation and checkpointing
  generation.py                   # sampling and text generation
  checkpoint.py                   # checkpoint serialization and restore
  web_runtime.py                  # local Shakespeare passage search
  cli.py                           # train / generate / chat / serve commands
api/generate.py                    # hosted app's fully offline chat endpoint
docs/FAI_SOURCE_DYING_EMBER.txt    # verbatim original source document
tests/                             # unit tests
public/                            # website UI, icon, styles, and browser script
```

## Current limitations

The current website runtime is intentionally honest about its limits: it can handle common chat patterns and supported local tasks, but it is not yet a broad general-purpose AI. The GPT model is initialized from random weights and no trained conversational checkpoint is included. A capable from-scratch model requires a large, diverse, legally usable dataset, substantial accelerator compute, evaluation, and safety work.

The Dying Ember source warns that messages may be incomplete, joking, false, exaggerated, or contradictory. Reconstruction claims must remain tied to actual evidence. The original document is preserved as-is; any future archive ingestion should add evidence without modifying that source.

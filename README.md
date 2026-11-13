# Neuron Forge

**An offline AI research project organized around the original Project Dying Ember FAI source document.** Neuron Forge contains a from-scratch decoder-only Transformer using RoPE, RMSNorm, SwiGLU, and scaled dot-product attention; a byte-level BPE tokenizer; CUDA mixed-precision training; checkpoint tools; browser-local conversation history; and a local Shakespeare reference library.

## Hosted website

Open **[Neuron Forge](https://neuron-forge-cyber-0ab0.vercel.app)**.

The hosted `/api/generate` endpoint is a hybrid offline runtime. It uses the trained 1,041,000-parameter Transformer through lightweight NumPy CPU inference for open-ended prompts, and preserves deterministic handlers for conversation recall, safe arithmetic, Shakespeare lookup, and source-grounded Dying Ember answers. It does not call Vercel AI Gateway or a third-party model provider. Recent conversation context and selected browser-local memories are sent to the project's own endpoint.

**Important distinction:** the live website now uses the 1M checkpoint for experimental open-ended text generation. The model is a small Shakespeare-heavy base language model, not an instruction-tuned assistant; its output can be incoherent or inaccurate and should not be treated as reliable factual reasoning.

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
- Conversation history saved in the browser (up to 12 chats) with cross-chat context retrieval.
- Bounded memory sanitization and lexical relevance ranking; obvious credential-bearing memory snippets are excluded from cross-chat context.
- Safe, constrained arithmetic parsing (no `eval`).
- Local Shakespeare corpus retrieval as optional reference material.
- Byte-level BPE tokenizer with a 256-byte base vocabulary and learned merges.
- NumPy CPU inference runtime for the deployed 1M checkpoint, without bundling PyTorch into the Vercel function.
- Decoder-only Transformer with RoPE, RMSNorm, SwiGLU feed-forward blocks, scaled dot-product attention, residual connections, and tied token embeddings.
- Training from random initialization using AdamW, warmup/cosine learning-rate scheduling, gradient accumulation, clipping, validation loss, checkpoints, and automatic CUDA mixed precision when supported.
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
  memory.py                       # bounded cross-chat memory ranking and sanitization
  config.py                       # model and training configuration
  tokenizer.py                    # byte-level BPE
  data.py                         # corpus encoding and batch sampling
  model.py                         # RoPE + RMSNorm + SwiGLU Transformer
  training.py                     # AdamW, mixed precision, evaluation and checkpointing
  generation.py                   # sampling and text generation
  checkpoint.py                   # checkpoint serialization and restore
  web_runtime.py                  # local Shakespeare passage search
  cli.py                           # train / generate / chat / serve commands
api/generate.py                    # hosted app's fully offline chat endpoint
docs/FAI_SOURCE_DYING_EMBER.txt    # verbatim original source document
tests/                             # unit tests
public/                            # website UI, icon, styles, and browser script
```

## Trained checkpoints

The new **1,041,000-parameter model** is committed in [`models/neuron-forge-1m/`](models/neuron-forge-1m/) and is wired into the live `/api/generate` endpoint for experimental open-ended prompts. Its training and limitations are documented in [`docs/NEURON_FORGE_1M_TRAINING.md`](docs/NEURON_FORGE_1M_TRAINING.md).

A separate **3.02B-parameter architecture blueprint** is recorded in [`configs/3b-architecture.json`](configs/3b-architecture.json) and [`docs/3B_SCALE_PLAN.md`](docs/3B_SCALE_PLAN.md). It has not been trained and has no checkpoint.

The earlier 258,528-parameter baseline and its continuation remain in [`models/neuron-forge-v1/`](models/neuron-forge-v1/). Historical results are recorded in [`docs/TRAINING_RESULTS_2026-11-13.md`](docs/TRAINING_RESULTS_2026-11-13.md).

## Current limitations

The live hybrid runtime is still experimental and limited. The 1M model is not instruction-tuned, and the current training corpus is Shakespeare-heavy plus the short FAI source document, which is not enough to produce a capable general-purpose assistant. The 3B architecture is only a blueprint; training it requires distributed/sharded training and dedicated high-memory CUDA hardware, and serving it would require a separate GPU inference service. The Vercel function intentionally bundles NumPy inference rather than PyTorch. A strong from-scratch model requires a much larger, diverse, legally usable corpus, substantial accelerator compute, instruction training, robust evaluation, and safety work. Browser memory is stored locally, but selected excerpts are sent to the project's own endpoint when a response is requested; it is not synced across devices.

The Dying Ember source warns that messages may be incomplete, joking, false, exaggerated, or contradictory. Reconstruction claims must remain tied to actual evidence. The original document is preserved as-is; any future archive ingestion should add evidence without modifying that source.

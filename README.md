# Neuron Forge

**An independent from-scratch general AI research project by Smiling Cat Labs.** Neuron Forge develops its own Transformer architecture, tokenizer, training pipeline, local chat runtime, and reference tools. It is not an FAI personality reconstruction and the 250M training configuration does not include FAI material.

## Live website

Open **[Neuron Forge](https://neuron-forge-cyber-0ab0.vercel.app)**.

The production website currently uses a small local conversation layer with browser-saved chat context, constrained arithmetic, and Shakespeare reference search. The experimental 1M checkpoint is not used for chat because its output did not meet basic quality standards.

## Main model target: 250M

A from-scratch architecture has been configured with **248,144,832 trainable parameters**. See [the exact model configuration](configs/250m.yaml), [architecture manifest](configs/250m-architecture.json), and [training plan](docs/250M_MODEL_PLAN.md).

| Setting | Value |
| --- | ---: |
| Parameters | 248,144,832 |
| Transformer layers | 29 |
| Attention heads | 16 |
| Embedding width | 832 |
| Context length | 1,024 |
| Position encoding | RoPE |
| Normalization | RMSNorm |
| Feed-forward | SwiGLU |
| Training | Random initialization, from scratch |

**Status: architecture configured; training not started.** The 250M workflow requires a self-hosted CUDA GPU runner and a separate, large, diverse, legally usable general-domain corpus at `data/general_corpus.txt`. It deliberately refuses to train on the old Shakespeare-only set or on FAI/Dying Ember project notes. The current CPU-only hosted runner and the current Vercel serverless function are not suitable for training or serving this model.

The model's parameter count is not a claim of intelligence. It must still be trained, instruction-tuned, and evaluated before it can answer like a dependable assistant. No 250M checkpoint exists yet.

## Other experiments

- The previous **1,041,000-parameter** checkpoint remains in the repository as a historical research experiment. It is disabled in live chat.
- The separate **3.02B** architecture blueprint remains in `configs/3b-architecture.json`; it has not been trained.

## Local development

Requirements: Python 3.10+ and a compatible PyTorch installation.

```bash
python -m venv .venv
# Windows PowerShell: .venv\\Scripts\\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev,training]"
pytest -q

# Inspect and count the 250M architecture without allocating its weights:
python - <<'PY'
from neuronforge_engine.config import load_config
cfg = load_config("configs/250m.yaml")
print(f"{cfg.model.parameter_count():,} trainable parameters")
PY
```

The 250M model's manual GPU training workflow is `.github/workflows/train-250m.yml`. It requires a runner labelled `self-hosted, linux, x64, gpu` and the separate general-domain corpus. It is intentionally not triggered by ordinary code pushes.

## Repository structure

```text
configs/250m.yaml                  # 250M training configuration
configs/250m-architecture.json    # exact architecture manifest
docs/250M_MODEL_PLAN.md            # hardware, data and training status
src/neuronforge_engine/model.py    # RoPE + RMSNorm + SwiGLU Transformer
src/neuronforge_engine/training.py # training loop with activation checkpointing
src/neuronforge_engine/offline_chat.py
src/neuronforge_engine/memory.py
api/generate.py                    # hosted local assistant endpoint
public/                            # website interface and assets
tests/                              # automated tests
```

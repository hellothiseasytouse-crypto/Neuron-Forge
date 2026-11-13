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

**Status: architecture configured; no completed 250M checkpoint yet.** The old self-hosted GitHub workflow remains available, and a resumable free-GPU Colab notebook has now been added. The notebook prepares its own attributed general-text corpus and saves checkpoints to Google Drive. The CPU-only hosted runner and the current Vercel serverless function are not suitable for training or serving this model.

The model's parameter count is not a claim of intelligence. It must still be trained, instruction-tuned, and evaluated before it can answer like a dependable assistant. No 250M checkpoint exists yet.

### Start the 250M training run

Open the [Neuron Forge 250M Google Colab notebook](https://colab.research.google.com/github/hellothiseasytouse-crypto/Neuron-Forge/blob/main/notebooks/train_250m_colab.ipynb). Select a free CUDA GPU if one is available, authorize Google Drive when prompted, and run the cells in order. It builds an English Wikipedia corpus with a source-attribution record, verifies the GPU, trains from random initialization, and saves optimizer-inclusive checkpoints to Drive every 250 steps so later sessions can resume.

Colab controls free GPU availability and session duration; neither is guaranteed. The 100,000-step target may require many sessions. Training is not considered complete until a checkpoint report records the final step and validation metrics. The live website does not load this model automatically.

## 40B-class model and training project

Neuron Forge includes a **40,000,905,216-parameter** decoder-only Transformer in [`configs/40b.yaml`](configs/40b.yaml). Its training system is separate from the 250M experiment.

| Setting | Value |
| --- | ---: |
| Parameters | 40,000,905,216 |
| Transformer layers | 49 |
| Attention heads | 64 |
| Embedding width | 8,192 |
| Context length | 2,048 |
| Vocabulary | 65,536 |
| Position encoding | RoPE |
| Normalization | RMSNorm |
| Feed-forward | SwiGLU |

The repository now includes a streaming dataset preparation pipeline, a tokenizer trained from the selected corpus, a BF16 PyTorch FSDP multi-GPU trainer, sharded model/optimizer checkpoints, a resumable manual GitHub Actions workflow, and a 40B training guide in [`docs/40B_TRAINING.md`](docs/40B_TRAINING.md).

**Training status: not started.** The data pipeline and trainer are code in GitHub; there is no trained 40B checkpoint yet. The current ChatGPT execution environment has CPU-only PyTorch and no CUDA devices, and GitHub's ordinary CI runners are CPU-only. The 40B workflow requires a separate self-hosted runner labelled `gpu` and `multi-gpu` with at least 16 visible BF16-capable GPUs. I have not launched a training run or claimed that weights exist.

The starter data manifest enables English Wikipedia and records attribution. It is not a complete general-purpose training corpus. The planned pretraining target is 800 billion tokens; more diverse, deduplicated and license-reviewed sources are needed to approach that target without excessive repetition. Large datasets and checkpoints stay on persistent training storage, not GitHub's source tree.

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

The 250M model also has a manual GitHub Actions workflow at `.github/workflows/train-250m.yml`, which requires a self-hosted CUDA GPU runner and a prepared general-domain corpus. It is intentionally not triggered by ordinary code pushes.

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

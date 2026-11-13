# Neuron Forge 250M model

## Architecture configured

The repository now contains a 250M-class decoder-only Transformer configuration.

| Setting | Value |
| --- | ---: |
| Exact trainable parameters | 248,144,832 |
| Transformer layers | 29 |
| Attention heads | 16 |
| Embedding width | 832 |
| Per-head width | 52 |
| Context length | 1,024 tokens |
| Vocabulary | 8,192 |
| SwiGLU hidden width | 2,224 |
| Position encoding | RoPE |
| Normalization | RMSNorm |
| Attention | Causal scaled dot-product attention |
| Token embeddings | Tied |

The exact count is calculated by `ModelConfig.parameter_count()` and checked by `tests/test_250m_config.py`.

## Training status

**Configured, not trained.** There are no 250M trained weights or checkpoint yet. The configuration uses random initialization and is designed for a dedicated CUDA GPU runner with activation checkpointing and gradient accumulation.

The current machine used for the repository's previous run was a CPU-only hosted runner. It cannot realistically train this model. The 250M job is manual-only and requires a self-hosted runner labelled `self-hosted, linux, x64, gpu`. It deliberately refuses to run on CPU.

## Data requirements

The old Shakespeare-heavy corpus is not suitable as the general-purpose training set. The 250M workflow requires a separate `data/general_corpus.txt` containing a large, diverse, legally usable general-domain corpus. FAI/Dying Ember project notes are not included in the 250M model's training data.

A 250M parameter count is not a capability guarantee. A useful language model needs a large token budget, high-quality and varied data, extensive optimization, instruction tuning, and evaluation. This first workflow is a training foundation, not a claim that Neuron Forge is already equivalent to a commercial assistant.

## Free-GPU training entry point

A resumable Google Colab notebook is available at
`notebooks/train_250m_colab.ipynb`. It checks for CUDA, builds a reproducible English Wikipedia corpus with source attribution stored alongside it, and persists model and optimizer checkpoints to Google Drive every 250 optimizer steps. On a later Colab session, rerunning the notebook resumes from the last saved step.

The notebook targets 100,000 optimizer steps with batch size 1 and gradient accumulation 8. Free GPU availability, device type, and runtime duration are controlled by Colab and are not guaranteed. A free notebook session may only complete a fraction of the target. Training remains incomplete until the final checkpoint report and validation metrics are verified.

## Deployment

The 250M model must not be loaded into the current Vercel serverless function. Its weights and inference memory need a dedicated GPU serving environment. After training, it must pass response-quality and instruction-following evaluations before being enabled in the website.

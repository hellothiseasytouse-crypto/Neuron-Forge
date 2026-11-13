# Neuron Forge 3B scale plan

## Architecture target

A separate 3B-scale architecture has been added as a blueprint in `configs/3b-architecture.json`.

| Setting | Value |
|---|---:|
| Approximate trainable parameters | 3,021,916,160 |
| Transformer layers | 60 |
| Attention heads | 16 |
| Embedding width | 2,048 |
| Context length | 2,048 tokens |
| Vocabulary | 384 |
| Feed-forward | SwiGLU |
| Normalization | RMSNorm |
| Position encoding | RoPE |

The parameter count is calculated from the current Neuron Forge model implementation, including tied token embeddings.

## Status

**Architecture blueprint only. The 3B model has not been trained and has no checkpoint.** The current GitHub Actions training runner is CPU-only, and the current trainer is single-process. It cannot train a model of this size. The model would require a distributed/sharded training implementation, a large licensed corpus, and high-memory CUDA accelerator infrastructure.

The live Vercel function is designed for the much smaller 1,041,000-parameter model. The 3B model would need a dedicated GPU inference service; it must not be bundled into the Vercel serverless function.

## Why this is not a one-click size change

A 3B parameter count does not itself create intelligence. Training data quality, token count, optimizer steps, hardware, instruction tuning, evaluation, and safety work all matter. The existing Shakespeare-heavy corpus is not enough to train a useful 3B general assistant.

# Architecture

Neuron Forge is a small decoder-only Transformer research project. It has no pretrained model weights and does not call a hosted model API.

## Data path

1. UTF-8 text is loaded from a local corpus.
2. `ByteBPETokenizer` learns deterministic byte-pair merges. The base vocabulary contains all 256 byte values, so arbitrary UTF-8 text can be represented.
3. Token IDs are split chronologically into training and validation sections.
4. Random windows form next-token prediction batches: inputs contain tokens `t..t+n`, targets contain `t+1..t+n+1`.

## Model path

The GPT implementation combines learned token and position embeddings, pre-normalized Transformer blocks, causal multi-head self-attention, GELU feed-forward networks, residual connections, final normalization, and a tied language-model output head. All trainable weights start from random initialization.

## Optimization

Training uses AdamW, warmup followed by cosine learning-rate decay, gradient accumulation, gradient clipping, periodic train/validation loss estimation, and atomic checkpoint writes. Checkpoints include model weights, tokenizer merges, configuration, step, and optimizer state.

## Generation

Generation repeatedly predicts the next token while cropping context to the model's maximum sequence length. Sampling supports temperature, top-k, nucleus (top-p), and repetition penalty.

## Interfaces

The CLI is the primary interface. An optional FastAPI server exposes `/health` and `/generate`, with a minimal browser chat page at `/`. The server binds to `127.0.0.1` by default.

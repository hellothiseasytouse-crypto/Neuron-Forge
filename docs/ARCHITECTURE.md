# Architecture

Neuron Forge currently has two distinct local systems:

1. An offline conversation prototype that powers the website chat without any external AI model API.
2. A from-scratch decoder-only Transformer training and inference stack.

The Transformer is not yet trained into a capable conversational model. The website therefore uses the inspectable offline runtime until a suitable checkpoint has been trained and evaluated. Do not describe the current website prototype as a general-purpose neural language model.

## Dying Ember source of truth

The original Dying Ember source document is preserved verbatim at `docs/FAI_SOURCE_DYING_EMBER.txt`. Do not edit or paraphrase that file when changing the code.

The local conversation runtime is organized around the source-defined systems:

- `PERSONALITY-AI`
- `FEELINGS-AI`
- `RULE-AI`
- `TRAITS-AI`

Together they form `FAI-AI`, as the source states. The source also states that the archive may be incomplete, joking, false, exaggerated, or contradictory, and that nothing is absolute without evidence. The runtime must not fabricate missing subject messages, memories, traits, or feelings. The source document is a mission specification; the underlying subject conversation archive must still be ingested as evidence before a reliable reconstruction can be made.

## Offline website chat

The hosted endpoint is `api/generate.py`. It calls only local Python modules:

1. `normalize_messages` validates a short user/assistant history.
2. `LocalConversationEngine` routes basic conversation, in-session recall, simple arithmetic, FAI-source questions, and requests tied to local source material.
3. `ShakespeareCorpus` optionally retrieves local passages when a query matches the bundled works.
4. The endpoint returns the local response and marks the mode as `offline`.

There is no Vercel AI Gateway, remote inference endpoint, API key, or external language-model call in this chat path. The runtime uses no profanity filter. Arithmetic is parsed through an explicit AST allowlist and never uses Python `eval`.

This runtime is intentionally limited and deterministic. It is not a replacement for a trained language model and should say so when a request exceeds its capabilities.

## From-scratch model data path

1. UTF-8 text is loaded from a local corpus.
2. `ByteBPETokenizer` learns deterministic byte-pair merges. The base vocabulary contains all 256 byte values, so arbitrary UTF-8 text can be represented.
3. Token IDs are split into training and validation sections.
4. Random windows form next-token prediction batches: inputs contain tokens `t..t+n`, targets contain `t+1..t+n+1`.

Only use data that is authorised for training. Keep source, license, language, and preprocessing metadata.

## Transformer model path

The GPT implementation combines learned token and position embeddings, pre-normalized Transformer blocks, causal multi-head self-attention, GELU feed-forward networks, residual connections, final normalization, and a tied language-model output head. All trainable weights start from random initialization.

## Optimization

Training uses AdamW, warmup followed by cosine learning-rate decay, gradient accumulation, gradient clipping, periodic train/validation loss estimation, and atomic checkpoint writes. Checkpoints include model weights, tokenizer merges, configuration, step, and optimizer state.

The quick training profile is a pipeline smoke test only; it is not enough to create a useful assistant.

## Generation

Generation repeatedly predicts the next token while cropping context to the model's maximum sequence length. Sampling supports temperature, top-k, nucleus (top-p), and repetition penalty.

## Interfaces

The CLI is the primary model-training interface. An optional FastAPI server exposes `/health` and `/generate`, with a minimal browser chat page at `/`. The server binds to `127.0.0.1` by default. Do not expose it publicly without authentication, request limits, TLS, and deployment hardening.

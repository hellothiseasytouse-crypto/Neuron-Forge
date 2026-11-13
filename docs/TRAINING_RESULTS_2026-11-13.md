# Neuron Forge training results — 2026-11-13

## Outcome

A decoder-only Transformer was trained from random initialization and then continued from its saved model and AdamW optimizer state. No pretrained weights or third-party AI API were used.

- [Initial training run](https://github.com/hellothiseasytouse-crypto/Neuron-Forge/actions/runs/43425107017)
- [Continued training run](https://github.com/hellothiseasytouse-crypto/Neuron-Forge/actions/runs/43425646511)
- [Download the continued checkpoint artifact](https://github.com/hellothiseasytouse-crypto/Neuron-Forge/actions/runs/43425646511/artifacts/14037251722)

The artifact contains `checkpoints/best.pt`, `checkpoints/latest.pt`, the training log, a generated sample, the corpus manifest, and a JSON training report. GitHub Actions retains it for 30 days, through 2026-12-13.

## Model configuration

| Setting | Value |
|---|---:|
| Architecture | Decoder-only Transformer |
| Parameters | 258,528 |
| Layers | 2 |
| Attention heads | 4 |
| Embedding width | 96 |
| Context length | 96 tokens |
| Vocabulary | 384 byte-BPE tokens |
| Attention | RoPE + causal scaled dot-product attention |
| Normalization | RMSNorm |
| Feed-forward | SwiGLU |
| Training device | Hosted CPU runner |
| Precision | FP32 |

## Training results

The initial run trained through step 1,000. The continued run loaded that run's `latest.pt` checkpoint and optimizer state and trained through step 6,000.

| Metric | Initial run | Continued run |
|---|---:|---:|
| Total training step | 1,000 | 6,000 |
| Best checkpoint step | 900 | 4,250 |
| Best validation loss | 3.5870 | 3.2794 |

The first run began with a validation loss of 5.9623 at step 1. The best continued-run validation loss was 3.2794. The validation loss is next-token prediction loss on the held-out portion of this corpus; it is not a measure of general reasoning or assistant quality.

## Training corpus

The build script assembled 43 text files totalling 938,388 words (42 local Shakespeare source files plus the canonical Dying Ember project source). The source manifest is included in the artifact.

The Dying Ember file is a short project specification. It is **not** the missing archive of FAI's conversations. Its inclusion does not mean the model has reconstructed FAI or learned FAI's personality.

## Quality and limitations

This is a small experimental base language model. It has learned some character- and word-level patterns from its corpus, but its generated sample still contains malformed words and incoherent text. It is not a general-purpose assistant, is not ChatGPT-level, and is not yet wired into the hosted Neuron Forge chat endpoint.

The current live website continues to use its offline conversation prototype. Replacing that runtime requires a separate inference integration and much stronger training and evaluation. A useful FAI reconstruction additionally requires the actual, permitted source archive; it must not be fabricated from the project specification.

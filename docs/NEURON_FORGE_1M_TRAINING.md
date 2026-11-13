# One-million-parameter Neuron Forge model

## Training outcome

The model was trained from random initialization on the hosted CPU runner. No pretrained checkpoint and no hosted AI API were used.

- [Training workflow run](https://github.com/hellothiseasytouse-crypto/Neuron-Forge/actions/runs/43427571015)
- [Training commit](https://github.com/hellothiseasytouse-crypto/Neuron-Forge/commit/deacd367410febe6f7b8b3a521dac32495561af0)
- [Model folder](https://github.com/hellothiseasytouse-crypto/Neuron-Forge/tree/main/models/neuron-forge-1m)
- [Training report](../models/neuron-forge-1m/training-report.json)
- [Generated sample](../models/neuron-forge-1m/sample.txt)
- [Exported NumPy runtime weights](../models/neuron-forge-1m/runtime.npz)
- [Live website](https://neuron-forge-cyber-0ab0.vercel.app)

## Architecture and metrics

| Setting | Value |
|---|---:|
| Trainable parameters | 1,041,000 |
| Transformer layers | 2 |
| Attention heads | 4 |
| Embedding width | 200 |
| Context length | 96 tokens |
| Vocabulary | 384 byte-level BPE tokens |
| Attention | RoPE + causal scaled dot-product attention |
| Normalization | RMSNorm |
| Feed-forward | SwiGLU |
| Training device | CPU |
| Precision | FP32 |
| Training steps | 2,500 |
| Best checkpoint step | 2,000 |
| Best validation loss | 3.0058 |

The validation loss is next-token prediction loss on the held-out portion of this corpus. It is not a measure of general reasoning, factual accuracy, or assistant quality.

## Training data

The local corpus builder assembled 43 text files containing 938,388 words: 42 Shakespeare source files plus the short canonical Dying Ember project specification. The complete manifest is saved with the model.

The Dying Ember source document is not the missing FAI conversation archive. The model has not reconstructed FAI's memories or personality.

## Files committed to the repository

- models/neuron-forge-1m/best.pt — best validation checkpoint.
- models/neuron-forge-1m/latest.pt — final checkpoint, including optimizer state.
- models/neuron-forge-1m/training-report.json — verified architecture and metrics.
- models/neuron-forge-1m/sample.txt — generated sample.
- models/neuron-forge-1m/corpus-manifest.json — source-file manifest.

## Limitations

This is a small experimental base language model, not a ChatGPT-level assistant. Its generated sample is still inconsistent and incoherent. A larger parameter count does not automatically produce better reasoning or broad knowledge.

The live Neuron Forge website now loads the exported NumPy runtime and uses the checkpoint for experimental open-ended generation through `/api/generate`. Deterministic handlers continue to protect saved-chat recall, arithmetic, Shakespeare lookup, and the Dying Ember evidence boundary. This integration avoids bundling PyTorch into the Vercel function. The deployed model is not instruction-tuned, so its generated answers can be unreliable and incoherent; it should be treated as an experimental text generator, not a dependable general-purpose assistant. A FAI-focused reconstruction also requires the actual permitted FAI conversation archive; that data has not been provided here.

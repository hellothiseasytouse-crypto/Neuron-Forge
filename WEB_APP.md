# Neuron Forge website

The website is served from public/index.html. Its same-origin /api/generate Python Function loads a small character-level recurrent neural network from public/model.json.

## Hosted web model

The deployment build creates an original conversation corpus and trains the compact character RNN from random initialization using only the Python standard library. The model weights are generated during the build and packaged with the Python Function. Inference runs on the server; the visitor's computer only renders the website.

This is a small experimental web demo, not a knowledgeable general-purpose assistant. The full decoder-only GPT Transformer, byte-level BPE tokenizer, PyTorch training loop, checkpoints, and generation sampler remain in src/neuronforge_engine/ for larger training runs. Install the optional training extra to use the PyTorch engine locally.

## Hosting limits

The Vercel web demo runs short inference requests against a small bundled checkpoint. It does not provide persistent online training, user-uploaded datasets, GPU training, or durable user checkpoints. Those require a separate persistent training service and durable model storage.

The original starter files in src/neuronforge/ remain unchanged.

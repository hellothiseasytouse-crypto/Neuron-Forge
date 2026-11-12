# Neuron Forge website

The website is served from `public/index.html`. Its stylesheet and client behavior are separated into `public/assets/neuron-forge.css` and `public/assets/neuron-forge.js`. The app icon is `public/favicon.svg`, and the browser/app identity is in `public/site.webmanifest`.

## Hosted model path

The Vercel build runs `scripts/prepare_data.py` to assemble the local text files under `data/shakespeare/`, then `scripts/build_corpus_index.py` to create `public/shakespeare_index.json`.

The same-origin `/api/generate` Python Function loads this bundled index and ranks matching passages. The runtime does not call a third-party model, text API, or external corpus host. It returns excerpts from the local collection rather than hallucinating free-form answers.

## Repository training engine

The separate decoder-only GPT, byte-level BPE tokenizer, PyTorch training loop, checkpoint format, and generation sampler remain under `src/neuronforge_engine/`. Assemble the bundled text with `python scripts/prepare_data.py`, then point a training configuration at `data/processed/shakespeare_corpus.txt` for a local training experiment.

## Hosting limits

The hosted passage engine is not a trained general-purpose assistant and cannot answer questions that are not represented by the collection. Vercel serves indexed passages; persistent online training, user-uploaded corpora, GPU training, and durable checkpoints require a separate persistent training service and storage.

The original starter files in `src/neuronforge/` remain unchanged.

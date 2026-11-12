# Training guide

## 1. Install

Use Python 3.10 or newer and install a PyTorch build appropriate for your machine.

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

## 2. Prepare text

Place UTF-8 `.txt` files in `data/raw/`, or combine a directory of text files:

```bash
python scripts/prepare_corpus.py --input ./my-text-files --output data/processed/corpus.txt
```

Only use text you have permission to use. Remove secrets and personal data. Keep a record of source, license, language, and preprocessing for each dataset.

## 3. Run a smoke test

```bash
python scripts/prepare_data.py
neuronforge train --config configs/quick.yaml
neuronforge generate --checkpoint checkpoints/latest.pt --prompt "Hello"
```

The sample corpus is intentionally repetitive. It checks that the code runs; it is not enough to train a useful assistant.

## 4. Train on your corpus

Update the data path in `configs/base.yaml`, then run:

```bash
neuronforge train --config configs/base.yaml
```

For a longer run, increase `max_steps`, evaluate loss curves, and keep a separate copy of your corpus and checkpoints. To resume from the configured checkpoint, pass `--resume`.

## 5. Use the local browser interface

```bash
pip install -e ".[web]"
neuronforge serve --checkpoint checkpoints/latest.pt
```

Open `http://127.0.0.1:8000`. The default server listens on loopback only. Do not expose it publicly without adding authentication, request limits, TLS, and deployment hardening.

## Practical expectations

A tiny model trained on a small corpus learns local patterns, not broad world knowledge or reliable reasoning. Training a capable general-purpose foundation model requires very large, diverse, legally usable datasets, substantial accelerator compute, careful evaluation, and safety work. This repository supplies a working educational training stack; it does not make those resource requirements disappear.

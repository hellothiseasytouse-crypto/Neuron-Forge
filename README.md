# Neuron Forge

**An independent from-scratch AI research project by Smiling Cat Labs.** Neuron Forge explores language-model architecture, training, inference, local conversation context, and reference tools without calling a third-party model provider.

## Live website

Open **[Neuron Forge](https://neuron-forge-cyber-0ab0.vercel.app)**.

The live website currently runs a transparent local assistant layer. It supports conversation context, browser-saved chat recall, constrained arithmetic, and search over the bundled Shakespeare reference library. The interface and chat runtime are independent of the FAI project.

## Model status

The repository contains a **1,041,000-parameter Transformer** trained from random initialization. It is a small base language model, not an instruction-tuned assistant. Its raw completions were incoherent, so the checkpoint has been disabled in live chat rather than shown as if it were a useful answer.

A separate **3.02B-parameter architecture blueprint** is recorded in `configs/3b-architecture.json` and `docs/3B_SCALE_PLAN.md`. It has not been trained and there is no 3B checkpoint.

A genuinely capable from-scratch assistant requires a much larger and more diverse training corpus, substantially more accelerator compute, instruction tuning, evaluation, and safety work. A parameter count by itself does not create intelligence.

## Current live capabilities

- Local conversation handling and bounded chat context.
- Browser-local conversation history (up to 12 chats).
- Relevant context retrieval from chats saved in the current browser.
- Constrained arithmetic parsing without `eval()`.
- Search over the local Shakespeare reference collection.
- No Vercel AI Gateway or third-party model API in the chat path.

## Training and development

Requirements: Python 3.10+ and a compatible PyTorch installation.

```bash
python -m venv .venv
# Windows PowerShell: .venv\\Scripts\\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev,training]"

python scripts/prepare_data.py
neuronforge train --config configs/quick.yaml
```

The quick profile is only a pipeline smoke test. It will not produce a capable conversational model.

## Repository structure

```text
src/neuronforge_engine/
  offline_chat.py                 # local chat routing, memory and safe arithmetic
  memory.py                       # bounded conversation-memory selection
  config.py                       # model and training configuration
  tokenizer.py                    # byte-level BPE tokenizer
  model.py                         # RoPE + RMSNorm + SwiGLU Transformer
  training.py                     # training loop and evaluation
  generation.py                   # experimental raw text generation
  checkpoint.py                   # checkpoint serialization and restore
  web_runtime.py                  # local Shakespeare passage search
  cli.py                           # train / generate / chat / serve commands
api/generate.py                    # hosted local assistant endpoint
public/                            # website interface and assets
tests/                              # automated tests
```

The FAI source document remains an archived research artifact in the repository; it is not used to define Neuron Forge's identity or live conversation behavior.

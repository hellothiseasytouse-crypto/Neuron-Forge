# Neuron Forge 4B training

Neuron Forge now has a separate 4B-class training path. The configured decoder-only Transformer has **4,064,498,688 trainable parameters**, initialized randomly. It does not start with pretrained weights.

## Added files

- `configs/4b.yaml`: model and training configuration.
- `data/4b-sources.yaml`: source manifest; English Wikipedia is enabled as a starter source and FineWeb-Edu is disabled pending review.
- `scripts/train_4b_fsdp.py`: PyTorch FSDP distributed trainer with BF16, gradient checkpointing, sharded optimizer/model checkpoints and resume.
- `.github/workflows/train-4b.yml`: manual GitHub Actions workflow for a self-hosted multi-GPU runner.

The data-preparation script currently shared with the 40B path is `scripts/prepare_40b_data.py`; it accepts the 4B manifest and writes to `data/4b`. Large data shards and checkpoints must stay on persistent training storage, not Git.

## Hardware

The workflow requires at least 8 visible BF16-capable NVIDIA GPUs. Eight is a minimum software gate, not a guarantee that every eight-GPU host has enough memory or throughput. The BF16 model weights alone are about 7.6 GiB total; training additionally needs gradients, optimizer states, activations and communication buffers. Use a suitably provisioned multi-GPU server and test a short run before committing to a long training job.

The current ChatGPT execution environment has no CUDA GPU, and no training run has been launched from it. A GitHub workflow only executes when a matching self-hosted runner is online and authorized. Do not describe the model as trained until a real checkpoint and loss metrics exist.

## Data requirements

The configured full-run guard requires at least 20 billion prepared training tokens and targets 100 billion tokens. The enabled Wikipedia starter source will not meet that requirement by itself. Add only reviewed datasets whose terms permit the intended use. Preserve attribution, review privacy and safety, deduplicate, and maintain held-out validation data. FineWeb-Edu remains disabled until its dataset card, ODC-By attribution obligations, Common Crawl terms, privacy and content filtering are reviewed.

Prepare data on the training machine:

```bash
python -m pip install -e . --no-deps
python -m pip install "datasets>=3.0" "tokenizers>=0.21,<1" "pyyaml>=6.0" "numpy>=1.26,<3"
python scripts/prepare_40b_data.py --config data/4b-sources.yaml --output-dir data/4b
```

Run on a configured multi-GPU host:

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=8 scripts/train_4b_fsdp.py \
  --config configs/4b.yaml --data-dir data/4b \
  --checkpoint-dir checkpoints/neuron-forge-4b
```

The workflow can be dispatched manually once a matching self-hosted GPU runner is registered. A short `max_steps` override is only an infrastructure smoke test, not meaningful pretraining.

## After pretraining

A pretrained next-token model is not yet a polished assistant. Neuron Forge still needs instruction tuning, conversation tuning, held-out evaluation, safety testing, retrieval/tool integration, and an inference deployment that can serve its checkpoint. The 4B checkpoint must be evaluated before claiming assistant quality.


## Dataset quality and instruction tuning

The pretraining preparation script now normalizes whitespace and removes exact duplicate document bodies before assigning documents to train/validation. This prevents exact duplicate text from appearing in both splits. It is not semantic deduplication or a complete safety/legal review.

Do not describe the current starter corpus as sufficient or fully reviewed. The enabled English Wikipedia source has attribution and share-alike obligations; check the source terms and preserve source metadata before distributing trained weights or derived datasets. FineWeb-Edu remains disabled pending its separate license, provenance, privacy and content review. The configured 20B-token minimum is a guard, not a claim that the current manifest supplies that many tokens.

### Prepare instruction data

Create a locally reviewed JSONL file with one conversation per line:

\`\`\`json
{"messages":[{"role":"user","content":"Explain photosynthesis simply."},{"role":"assistant","content":"Photosynthesis is the process..."}]}
\`\`\`

Then, after the 4B tokenizer and pretrained checkpoint exist:

\`\`\`bash
python scripts/prepare_4b_sft.py --input data/reviewed-conversations.jsonl --tokenizer data/4b/tokenizer.json --output-dir data/4b-sft
torchrun --standalone --nnodes=1 --nproc_per_node=8 scripts/instruction_tune_4b_fsdp.py --config configs/4b-sft.yaml --data-dir data/4b-sft --pretrained-checkpoint checkpoints/neuron-forge-4b --checkpoint-dir checkpoints/neuron-forge-4b-sft
\`\`\`

The SFT data builder removes exact duplicate conversations, creates a deterministic held-out split, and masks user/system tokens so only assistant response tokens contribute to supervised loss. It does not itself certify the source data; the owner must review and document the source license, provenance, privacy and quality.

Both training stages append training/validation measurements to \`metrics.jsonl\` and save resumable distributed checkpoints. The instruction-tuning workflow is manual and also requires the same self-hosted multi-GPU runner plus persistent access to the pretrained checkpoint and prepared SFT shards.

**Testing milestone:** after SFT produces \`checkpoints/neuron-forge-4b-sft/latest.json\`, export/load the checkpoint into an inference runtime and run a held-out conversational evaluation suite. A falling validation loss alone is not proof of a reliable assistant. The current repository does not yet contain a production-ready 4B chat-serving endpoint or a completed evaluation scorecard.

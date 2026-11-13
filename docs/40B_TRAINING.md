# Neuron Forge 40B training

## What is in this repository

The 40B project is now more than a parameter-count file. It includes:

- `configs/40b.yaml`: the 40,000,905,216-parameter model and 800-billion-token training target.
- `data/40b-sources.yaml`: an enabled starter source with source/license metadata.
- `scripts/prepare_40b_data.py`: streaming dataset ingestion, a tokenizer trained from scratch, and train/validation token shards.
- `scripts/train_40b_fsdp.py`: BF16 multi-GPU FSDP pretraining and sharded model + optimizer checkpoints.
- `.github/workflows/train-40b.yml`: a manually started workflow for a configured self-hosted multi-GPU runner.

Large datasets, token shards, and checkpoints are deliberately not committed to Git. They belong on persistent storage attached to the training machine.

## Data and knowledge

The starter source is the English Wikipedia dump subset `20231101.en` through the Hugging Face dataset `wikimedia/wikipedia`. The pipeline preserves the article title and URL and writes a dataset attribution manifest. Wikipedia is only a starting source; it is not "everything" and is not enough by itself for a well-rounded 40B assistant.

Before enabling another source, check its dataset card, upstream terms, and whether the intended model training and weight distribution are allowed. Add a new entry to `data/40b-sources.yaml` with dataset ID, configuration, text field, declared license, attribution, and dataset card URL. Only enable material that has been reviewed. Do not put private chats, passwords, personal records, or copyrighted books/music into the corpus without the right to use them.

A 40B model needs a large, diverse, deduplicated, quality-filtered corpus. The configured target is 800 billion tokens as an initial compute/data planning target, not a guarantee of quality. Training repeatedly on a much smaller corpus can overfit badly. Source curation, deduplication, safety filtering, evaluation data, and instruction-tuning data are still required.

## Hardware reality

This is a large distributed training job. The supplied FSDP implementation refuses to run with fewer than 16 GPUs and requires BF16-capable CUDA devices. A practical run needs a multi-GPU server or cluster, fast interconnect/storage, and persistent checkpoint storage. The exact safe configuration depends on GPU memory, communication bandwidth, and PyTorch/CUDA versions.

The BF16 weights alone require about 74.5 GiB in total. Training also needs gradients, AdamW states, activations, temporary all-gather buffers, and data-loading memory. Sixteen GPUs is a software guardrail for this initial FSDP path, not a promise that every 16-GPU machine has enough memory or throughput.

A normal GitHub-hosted CPU runner, a laptop CPU, or a free Colab T4 cannot train this model. The current chat execution environment has CPU-only PyTorch and no CUDA devices, so no 40B training step has run here.

## Prepare data on the training machine

Install a CUDA-compatible PyTorch build for the machine first. Then install the project and dataset tools:

```bash
python -m pip install -e . --no-deps
python -m pip install "datasets>=3.0" "tokenizers>=0.21,<1" "pyyaml>=6.0" "numpy>=1.26,<3"
python scripts/prepare_40b_data.py --config data/40b-sources.yaml --output-dir data/40b
```

To rebuild an existing local dataset, add `--overwrite`. The script streams the configured datasets; it does not place dataset files in GitHub. It writes `train.bin`, `validation.bin`, `tokenizer.json`, and `dataset-manifest.json`.

## Train across GPUs

On one server with at least 16 visible BF16-capable CUDA GPUs:

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=16 scripts/train_40b_fsdp.py \
  --config configs/40b.yaml \
  --data-dir data/40b \
  --checkpoint-dir checkpoints/neuron-forge-40b
```

The trainer computes the update count from `target_tokens` and the actual world size. It saves sharded model and optimizer checkpoints at evaluation intervals and resumes from `latest.json` by default. Checkpoints must remain on persistent shared storage if the machine can be replaced.

For an explicitly limited test on a properly provisioned cluster, pass `--max-steps N`; such a run is only a smoke test and must not be described as a finished 40B model.

## What comes after pretraining

Pretraining teaches next-token prediction. It does not automatically create a dependable assistant. After pretraining, Neuron Forge still needs:

1. Evaluation on held-out knowledge, reasoning, coding, math, multilingual, and safety benchmarks.
2. Supervised instruction tuning on reviewed instruction/response examples.
3. Preference/safety tuning and red-team evaluation.
4. Retrieval and tool-use integration for current information and calculations.
5. Quantization and a separately hosted inference service before the website can serve the weights.

A checkpoint is only considered trained when its saved step count, token count, validation loss, and evaluation report are recorded. A successful workflow or an architecture config alone is not proof of model quality.

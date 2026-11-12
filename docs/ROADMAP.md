# Roadmap

- [x] Foundation source layout supplied in the initial ZIP
- [x] Byte-level BPE tokenizer
- [x] Dataset encoding, chronological train/validation split, and batch sampling
- [x] Causal decoder-only Transformer with residual MLP blocks
- [x] AdamW training, warmup/cosine schedule, gradient clipping, evaluation, and checkpoints
- [x] Text generation with temperature, top-k, top-p, and repetition penalty
- [x] CLI training, generation, and terminal chat
- [x] Optional local FastAPI endpoint and browser UI
- [x] Unit tests and continuous integration
- [ ] Add a tokenizer benchmark suite and streaming data loader for large corpora
- [ ] Add experiment tracking and reproducible multi-run comparisons
- [ ] Add larger-scale distributed training after hardware and dataset planning
- [ ] Add instruction fine-tuning and preference evaluation as separate, documented stages
- [ ] Add security review before any public deployment

The checked items describe repository implementation, not a claim that a capable foundation model has already been trained.

# Data, privacy, and safety notes

- Train only on content you own, have permission to use, or that is licensed for the intended use.
- Do not put passwords, API keys, private messages, personal records, or confidential documents into the training corpus.
- Keep raw datasets out of Git. The repository ignores `data/raw/` and model checkpoints by default.
- Track dataset source, license, date acquired, language, filtering, and any transformations in a local manifest.
- A model can memorize fragments of training text. Treat generated output as potentially sensitive and inspect it before sharing.
- The optional API is intended for local experimentation and binds to `127.0.0.1` by default. It has no user authentication, rate limiting, or multi-user isolation.
- Do not describe this early research model as safe, factual, unbiased, or production-ready without evidence from appropriate evaluations.

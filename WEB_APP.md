# Neuron Forge hosted website

The website is served from `public/index.html`. The same-origin `/api/generate` Python Function loads the project's own PyTorch checkpoint and performs local model inference on the server.

## Deployment

The Vercel build command in `vercel.json` installs the CPU-only PyTorch dependencies, creates the tiny example corpus, and runs the existing from-scratch training pipeline with `configs/quick.yaml`. The resulting `checkpoints/latest.pt` file is bundled into the Python Function.

The hosted checkpoint is deliberately a small pipeline demonstration (100 training steps on a repetitive test corpus). It is not a capable general-purpose assistant. For meaningful model quality, replace the demo corpus with licensed training data and run a properly resourced training job.

## Important hosting behavior

Vercel Functions are serverless and have bounded execution time and memory. The hosted API is for short inference requests against the prebuilt demo checkpoint; it does not provide persistent online training, background jobs, or durable user-uploaded checkpoints. Those require a separate persistent training service and durable model storage.

The original files in `src/neuronforge/` remain unchanged.

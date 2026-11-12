"""Writes a tiny sample corpus to data/raw/input.txt so training has something to read.

Replace this with your own text file for real results.
"""
from pathlib import Path

SAMPLE = (
    "Neuron-Forge is a language model built from scratch. "
    "It learns to predict the next character from the ones before it. "
) * 200

out = Path("data/raw/input.txt")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(SAMPLE, encoding="utf-8")
print(f"Wrote {len(SAMPLE)} characters to {out}")

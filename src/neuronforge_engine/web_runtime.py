"""Dependency-free character RNN for the hosted Neuron Forge web demo."""
from __future__ import annotations
import json, math, random
from pathlib import Path
from typing import Any

class CharacterRNN:
    def __init__(self, state: dict[str, Any]):
        self.vocab = state["vocab"]
        self.index = {char: i for i, char in enumerate(self.vocab)}
        self.hidden_size = int(state["hidden_size"])
        self.wxh, self.whh = state["wxh"], state["whh"]
        self.bh, self.why, self.by = state["bh"], state["why"], state["by"]
        self.training_steps = int(state.get("training_steps", 0))
        self.training_chars = int(state.get("training_chars", 0))
        v, h = len(self.vocab), self.hidden_size
        self.parameter_count = v*h + h*h + h + v*h + v

    @classmethod
    def load(cls, path: str | Path) -> "CharacterRNN":
        model_path = Path(path)
        if not model_path.is_file():
            raise FileNotFoundError(f"Web model not found: {model_path}")
        with model_path.open("r", encoding="utf-8") as handle:
            return cls(json.load(handle))

    def _step(self, char_id: int, hidden: list[float]):
        h_size = self.hidden_size
        next_hidden = [0.0] * h_size
        input_row = self.wxh[char_id]
        for j in range(h_size):
            value = self.bh[j] + input_row[j]
            row = self.whh[j]
            for k in range(h_size):
                value += row[k] * hidden[k]
            next_hidden[j] = math.tanh(value)
        logits = [0.0] * len(self.vocab)
        for token_id, row in enumerate(self.why):
            value = self.by[token_id]
            for j in range(h_size):
                value += row[j] * next_hidden[j]
            logits[token_id] = value
        return next_hidden, logits

    def generate(self, prompt: str, max_new_tokens: int = 120,
                 temperature: float = 0.8, top_k: int = 5) -> str:
        temperature = max(0.15, min(float(temperature), 1.5))
        top_k = max(1, min(int(top_k), len(self.vocab)))
        fallback = self.index.get(" ", 0)
        hidden = [0.0] * self.hidden_size
        next_logits = []
        for char in prompt:
            hidden, next_logits = self._step(self.index.get(char, fallback), hidden)
        if not next_logits:
            hidden, next_logits = self._step(fallback, hidden)

        rng = random.Random()
        output = []
        for _ in range(max_new_tokens):
            ranked = sorted(range(len(next_logits)),
                            key=lambda i: next_logits[i], reverse=True)[:top_k]
            peak = max(next_logits[i] for i in ranked)
            weights = [math.exp((next_logits[i] - peak) / temperature) for i in ranked]
            choice = rng.choices(ranked, weights=weights, k=1)[0]
            char = self.vocab[choice]
            output.append(char)
            hidden, next_logits = self._step(choice, hidden)
            current = "".join(output)
            if current.endswith("\n\n") or "\nUser:" in current:
                break
        result = "".join(output)
        if "\nUser:" in result:
            result = result.split("\nUser:", 1)[0]
        if "\n\n" in result:
            result = result.split("\n\n", 1)[0]
        return result.strip()

def load_web_model(path: str | Path) -> CharacterRNN:
    return CharacterRNN.load(path)

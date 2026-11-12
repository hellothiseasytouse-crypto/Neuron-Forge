"""Train a compact character RNN with the Python standard library only."""
from __future__ import annotations
import json, math, random
from pathlib import Path

CORPUS = Path("data/raw/input.txt")
OUTPUT = Path("public/model.json")
HIDDEN, SEQUENCE, STEPS = 32, 36, 900
LEARNING_RATE, SEED = 0.08, 20261112

def zeros(rows, cols):
    return [[0.0] * cols for _ in range(rows)]

def random_matrix(rows, cols, scale, rng):
    return [[rng.uniform(-scale, scale) for _ in range(cols)] for _ in range(rows)]

def softmax(values):
    peak = max(values)
    exps = [math.exp(value - peak) for value in values]
    total = sum(exps)
    return [value / total for value in exps]

def train():
    if not CORPUS.is_file():
        raise FileNotFoundError(f"Training corpus not found: {CORPUS}")
    text = CORPUS.read_text(encoding="utf-8")
    if len(text) < SEQUENCE * 2:
        raise ValueError("Training corpus is too small")
    vocab = sorted(set(text))
    lookup = {char: i for i, char in enumerate(vocab)}
    ids = [lookup[char] for char in text]
    v, h = len(vocab), HIDDEN
    rng = random.Random(SEED)
    wxh = random_matrix(v, h, 0.08, rng)
    whh = random_matrix(h, h, 0.05, rng)
    bh = [0.0] * h
    why = random_matrix(v, h, 0.08, rng)
    by = [0.0] * v
    print(f"Training web RNN from random initialization: vocab={v}, hidden={h}, steps={STEPS}")

    for step in range(1, STEPS + 1):
        start = rng.randrange(0, len(ids) - SEQUENCE - 1)
        inputs = ids[start:start + SEQUENCE]
        targets = ids[start + 1:start + SEQUENCE + 1]
        hidden_states = [[0.0] * h]
        output_grads = []
        hidden = hidden_states[0]
        loss = 0.0

        for token_id, target_id in zip(inputs, targets):
            next_hidden = [0.0] * h
            input_row = wxh[token_id]
            for j in range(h):
                value = bh[j] + input_row[j]
                row = whh[j]
                for k in range(h):
                    value += row[k] * hidden[k]
                next_hidden[j] = math.tanh(value)
            logits = [0.0] * v
            for token, row in enumerate(why):
                value = by[token]
                for j in range(h):
                    value += row[j] * next_hidden[j]
                logits[token] = value
            probs = softmax(logits)
            loss -= math.log(max(probs[target_id], 1e-12))
            dlogits = probs[:]
            dlogits[target_id] -= 1.0
            output_grads.append(dlogits)
            hidden_states.append(next_hidden)
            hidden = next_hidden

        gwhy = zeros(v, h)
        gby = [0.0] * v
        dh_states = [[0.0] * h for _ in range(SEQUENCE)]
        for t, dlogits in enumerate(output_grads):
            hidden_t = hidden_states[t + 1]
            for token in range(v):
                grad = dlogits[token]
                gby[token] += grad
                row_grad = gwhy[token]
                for j in range(h):
                    row_grad[j] += grad * hidden_t[j]
            dh = dh_states[t]
            for j in range(h):
                for token in range(v):
                    dh[j] += why[token][j] * dlogits[token]

        gwxh = zeros(v, h)
        gwhh = zeros(h, h)
        gbh = [0.0] * h
        dh_next = [0.0] * h
        for t in range(SEQUENCE - 1, -1, -1):
            hidden_t = hidden_states[t + 1]
            hidden_prev = hidden_states[t]
            dh_raw = [0.0] * h
            for j in range(h):
                total = dh_states[t][j] + dh_next[j]
                dh_raw[j] = total * (1.0 - hidden_t[j] * hidden_t[j])
                gbh[j] += dh_raw[j]
            token_id = inputs[t]
            gx_row = gwxh[token_id]
            for j in range(h):
                gx_row[j] += dh_raw[j]
            dh_prev = [0.0] * h
            for j in range(h):
                grad_value = dh_raw[j]
                row_grad = gwhh[j]
                row = whh[j]
                for k in range(h):
                    row_grad[k] += grad_value * hidden_prev[k]
                    dh_prev[k] += row[k] * grad_value
            dh_next = dh_prev

        scale = 1.0 / SEQUENCE
        for matrix, grad in ((wxh, gwxh), (whh, gwhh), (why, gwhy)):
            for i, row in enumerate(matrix):
                grad_row = grad[i]
                for j in range(len(row)):
                    g = max(-2.0, min(2.0, grad_row[j] * scale))
                    row[j] -= LEARNING_RATE * g
        for vector, grad in ((bh, gbh), (by, gby)):
            for i in range(len(vector)):
                g = max(-2.0, min(2.0, grad[i] * scale))
                vector[i] -= LEARNING_RATE * g
        if step % 150 == 0:
            print(f"step {step:4d}/{STEPS} | loss {loss / SEQUENCE:.4f}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "format_version": 1, "model_type": "character-rnn", "vocab": vocab,
        "hidden_size": h, "wxh": wxh, "whh": whh, "bh": bh,
        "why": why, "by": by, "training_steps": STEPS,
        "training_chars": len(ids), "seed": SEED,
    }
    OUTPUT.write_text(json.dumps(state, separators=(",", ":")), encoding="utf-8")
    print(f"Saved {OUTPUT} ({OUTPUT.stat().st_size:,} bytes)")

if __name__ == "__main__":
    train()

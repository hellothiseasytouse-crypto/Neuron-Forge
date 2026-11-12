"""Autoregressive sampling utilities."""
import torch

from .checkpoint import load_model
from .model import GPT
from .tokenizer import ByteBPETokenizer


def sample_next_token(logits: torch.Tensor, temperature: float = 1.0,
                      top_k: int | None = None, top_p: float = 1.0,
                      repetition_penalty: float = 1.0,
                      recent_ids: torch.Tensor | None = None) -> torch.Tensor:
    if temperature < 0:
        raise ValueError("temperature must be >= 0")
    if not 0.0 < top_p <= 1.0:
        raise ValueError("top_p must be in (0, 1]")
    if repetition_penalty <= 0:
        raise ValueError("repetition_penalty must be positive")
    scores = logits.clone()
    if recent_ids is not None and repetition_penalty != 1.0:
        for token_id in torch.unique(recent_ids):
            value = scores[:, token_id]
            scores[:, token_id] = torch.where(value < 0, value * repetition_penalty,
                                                value / repetition_penalty)
    if temperature == 0:
        return torch.argmax(scores, dim=-1, keepdim=True)
    scores = scores / temperature
    if top_k is not None:
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        k = min(top_k, scores.size(-1))
        threshold = torch.topk(scores, k, dim=-1).values[:, -1, None]
        scores = scores.masked_fill(scores < threshold, torch.finfo(scores.dtype).min)
    if top_p < 1.0:
        sorted_scores, sorted_indices = torch.sort(scores, descending=True, dim=-1)
        sorted_probs = torch.softmax(sorted_scores, dim=-1)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        remove = cumulative - sorted_probs >= top_p
        sorted_scores = sorted_scores.masked_fill(remove, torch.finfo(sorted_scores.dtype).min)
        filtered = torch.full_like(scores, torch.finfo(scores.dtype).min)
        scores = filtered.scatter(1, sorted_indices, sorted_scores)
    probabilities = torch.softmax(scores, dim=-1)
    return torch.multinomial(probabilities, num_samples=1)


@torch.inference_mode()
def generate_from_model(model: GPT, tokenizer: ByteBPETokenizer, prompt: str,
                        max_new_tokens: int = 200, temperature: float = 0.8,
                        top_k: int | None = 40, top_p: float = 0.95,
                        repetition_penalty: float = 1.05,
                        device: torch.device | str | None = None) -> str:
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    model.eval()
    target_device = device or next(model.parameters()).device
    start = tokenizer.encode(prompt) if prompt else [10]
    idx = torch.tensor([start], dtype=torch.long, device=target_device)
    for _ in range(max_new_tokens):
        context = idx[:, -model.config.block_size:]
        logits, _ = model(context)
        next_id = sample_next_token(logits[:, -1, :], temperature, top_k, top_p,
                                    repetition_penalty, context)
        idx = torch.cat((idx, next_id), dim=1)
    return tokenizer.decode(idx[0].tolist())


def generate(checkpoint: str, prompt: str, max_new_tokens: int = 200,
             temperature: float = 0.8, top_k: int | None = 40,
             top_p: float = 0.95, repetition_penalty: float = 1.05,
             device: str = "cpu") -> str:
    model, tokenizer, _ = load_model(checkpoint, device=device)
    return generate_from_model(model, tokenizer, prompt, max_new_tokens, temperature,
                               top_k, top_p, repetition_penalty, device)

"""Local conversation-memory selection and input sanitization.

Conversation memory is supplied by the browser from its own local storage. The
server does not claim durable server-side storage and never calls a remote model.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Any

TOKEN_RE = re.compile(r"[a-z0-9]+(?:['’][a-z0-9]+)?", re.I)
STOP_WORDS = {
    "a", "about", "an", "and", "are", "as", "at", "be", "but", "by", "can",
    "did", "do", "for", "from", "had", "has", "have", "how", "i", "in", "is",
    "it", "me", "my", "of", "on", "or", "our", "please", "that", "the", "this",
    "to", "was", "we", "what", "when", "where", "which", "who", "with", "you",
    "your", "remember", "recall", "tell",
}
SENSITIVE_RE = re.compile(
    r"\b(password|passcode|api[ -]?key|secret|access token|refresh token|"
    r"credit card|cvv|social security|one[- ]time password|otp)\b",
    re.I,
)


def normalize_memory_entries(raw: Any, max_entries: int = 24, max_chars: int = 12000) -> list[dict[str, str]]:
    """Accept only bounded user/assistant text records; discard obvious credentials."""
    if not isinstance(raw, list):
        return []
    cleaned: list[dict[str, str]] = []
    total = 0
    seen: set[tuple[str, str]] = set()
    for item in raw[-max_entries:]:
        if not isinstance(item, dict):
            continue
        role, content = item.get("role"), item.get("content")
        if role not in {"user", "assistant"} or not isinstance(content, str):
            continue
        content = " ".join(content.split()).strip()[:1200]
        if not content or SENSITIVE_RE.search(content):
            continue
        key = (role, content.casefold())
        if key in seen:
            continue
        if total + len(content) > max_chars:
            continue
        seen.add(key)
        cleaned.append({"role": role, "content": content})
        total += len(content)
    return cleaned


def select_relevant_memories(query: str, memories: list[dict[str, str]], limit: int = 6) -> list[dict[str, str]]:
    """Rank saved turns by lexical relevance and recency without inventing facts."""
    if not memories or limit < 1:
        return []
    query_tokens = {
        token for token in TOKEN_RE.findall(query.lower().replace("’", "'"))
        if token not in STOP_WORDS and len(token) > 1
    }
    ranked: list[tuple[float, int, dict[str, str]]] = []
    for index, memory in enumerate(memories):
        tokens = [t for t in TOKEN_RE.findall(memory["content"].lower().replace("’", "'")) if t not in STOP_WORDS]
        counts = Counter(tokens)
        overlap = sum(min(count, 3) for token, count in counts.items() if token in query_tokens)
        coverage = len(query_tokens & set(counts)) / max(1, len(query_tokens))
        recency = (index + 1) / len(memories)
        score = (overlap * 1.5) + (coverage * 2.0) + (recency * 0.35)
        if not query_tokens:
            score = recency
        if score > 0:
            ranked.append((score, index, memory))
    ranked.sort(key=lambda row: (row[0], row[1]), reverse=True)
    selected = [row[2] for row in ranked[:limit]]
    selected.reverse()
    return selected

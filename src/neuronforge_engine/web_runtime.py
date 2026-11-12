"""Local Shakespeare corpus retrieval engine for Neuron Forge's hosted demo."""
from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

WORD_RE = re.compile(r"[a-z0-9]+(?:['’][a-z0-9]+)?", re.IGNORECASE)
STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "had", "has", "have", "he", "her", "hers", "him", "his", "how", "i",
    "if", "in", "into", "is", "it", "its", "me", "my", "of", "on", "or",
    "our", "she", "so", "that", "the", "their", "them", "there", "these",
    "they", "this", "those", "to", "us", "was", "we", "were", "what",
    "when", "where", "which", "who", "will", "with", "you", "your",
}

class ShakespeareCorpus:
    """Keyword-ranked passage engine backed only by files in this repository."""

    def __init__(self, payload: dict[str, Any]):
        self.works = payload.get("works", [])
        self.passages = payload.get("passages", [])
        self.word_count = int(payload.get("word_count", 0))
        self.index: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.document_frequency: Counter[str] = Counter()
        for i, passage in enumerate(self.passages):
            tokens = self._tokens(passage.get("text", ""))
            counts = Counter(tokens)
            for token, count in counts.items():
                self.index[token].append((i, min(count, 5)))
                self.document_frequency[token] += 1
        self.total_passages = max(1, len(self.passages))

    @staticmethod
    def _tokens(text: str) -> list[str]:
        return [token.lower().replace("’", "'") for token in WORD_RE.findall(text)]

    @classmethod
    def load(cls, path: str | Path) -> "ShakespeareCorpus":
        model_path = Path(path)
        if not model_path.is_file():
            raise FileNotFoundError(f"Shakespeare corpus index not found: {model_path}")
        with model_path.open("r", encoding="utf-8") as handle:
            return cls(json.load(handle))

    def describe(self) -> dict[str, Any]:
        return {
            "model": "Neuron Forge · Shakespeare Corpus",
            "architecture": "local corpus search and passage ranking",
            "works": len(self.works),
            "passages": len(self.passages),
            "training_words": self.word_count,
            "trained_from_local_files": True,
        }

    def search(self, prompt: str, limit: int = 3) -> list[dict[str, Any]]:
        raw_tokens = self._tokens(prompt)
        query = list(dict.fromkeys(token for token in raw_tokens if token not in STOP_WORDS))
        if not query:
            query = list(dict.fromkeys(raw_tokens))
        if not query:
            return []

        scores: dict[int, float] = defaultdict(float)
        query_set = set(query)
        for token in query:
            idf = math.log(1.0 + self.total_passages / (1.0 + self.document_frequency.get(token, 0)))
            for passage_id, frequency in self.index.get(token, []):
                scores[passage_id] += idf * (1.0 + math.log(frequency))

        normalized_prompt = " ".join(raw_tokens)
        ranked = []
        for passage_id, score in scores.items():
            passage = self.passages[passage_id]
            text = passage.get("text", "")
            lowered = " ".join(self._tokens(text))
            matched = sum(1 for token in query_set if token in lowered)
            coverage = matched / max(1, len(query_set))
            score *= 0.6 + coverage
            if len(normalized_prompt) >= 12 and normalized_prompt in lowered:
                score *= 2.5
            title_tokens = set(self._tokens(passage.get("work", "")))
            title_overlap = len(query_set & title_tokens)
            score += title_overlap * 3.0
            ranked.append((score, passage_id))
        ranked.sort(reverse=True)

        results = []
        used = set()
        for score, passage_id in ranked:
            passage = self.passages[passage_id]
            work = passage.get("work", "Unknown work")
            signature = (work, passage.get("text", "")[:80])
            if signature in used:
                continue
            used.add(signature)
            text = passage.get("text", "").strip()
            if not text:
                continue
            results.append({
                "work": work,
                "category": passage.get("category", "work"),
                "text": text,
                "score": round(score, 3),
            })
            if len(results) >= limit:
                break
        return results

    def generate(self, prompt: str, max_results: int = 2) -> str:
        results = self.search(prompt, limit=max(1, min(int(max_results), 3)))
        if not results:
            return (
                "I couldn't match that wording in the local Shakespeare collection. "
                "Try a character, play, sonnet, phrase, or theme such as ambition, love, "
                "jealousy, mercy, time, or magic."
            )
        blocks = []
        for item in results:
            words = item["text"].split()
            excerpt = " ".join(words[:115])
            if len(words) > 115:
                excerpt += " …"
            blocks.append(f"From {item['work']}:\n{excerpt}")
        return "\n\n".join(blocks)

def load_web_model(path: str | Path) -> ShakespeareCorpus:
    return ShakespeareCorpus.load(path)

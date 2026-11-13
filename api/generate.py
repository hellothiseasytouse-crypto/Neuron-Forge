"""Vercel Python Function for Neuron Forge's local assistant runtime."""
from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neuronforge_engine.offline_chat import LocalConversationEngine  # noqa: E402
from neuronforge_engine.memory import normalize_memory_entries  # noqa: E402
from neuronforge_engine.web_runtime import load_web_model  # noqa: E402

CORPUS_INDEX = ROOT / "public" / "shakespeare_index.json"
_engine = None
_corpus = None


def get_engine() -> LocalConversationEngine:
    global _engine
    if _engine is None:
        _engine = LocalConversationEngine()
    return _engine


def get_corpus():
    global _corpus
    if _corpus is None:
        _corpus = load_web_model(CORPUS_INDEX)
    return _corpus


def send_json(handler, status: int, payload: dict):
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Cache-Control", "no-store")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def should_search_shakespeare(message: str, corpus) -> bool:
    lowered = message.lower()
    library_terms = (
        "shakespeare", "sonnet", "sonnets", "passage", "quote", "quotation",
        "line from", "which play", "what play", "act ", "scene ", "monologue",
        "character in", "speech in",
    )
    if any(term in lowered for term in library_terms):
        return True
    return any(
        isinstance(work, dict)
        and isinstance(work.get("title"), str)
        and work["title"].lower() in lowered
        for work in corpus.works
    )


def normalize_messages(payload: dict) -> list[dict[str, str]]:
    raw = payload.get("messages")
    if not isinstance(raw, list):
        prompt = payload.get("prompt", "")
        if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 2000:
            raise ValueError("Prompt must contain 1–2,000 characters")
        return [{"role": "user", "content": prompt.strip()}]

    cleaned: list[dict[str, str]] = []
    total_chars = 0
    for item in raw[-24:]:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        content = item.get("content")
        if role not in {"user", "assistant"} or not isinstance(content, str):
            continue
        content = content.strip()
        if not content:
            continue
        content = content[:2000]
        if total_chars + len(content) > 28000:
            break
        cleaned.append({"role": role, "content": content})
        total_chars += len(content)

    if not cleaned or not any(item["role"] == "user" for item in cleaned):
        raise ValueError("At least one user message is required")
    return cleaned


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.split("?", 1)[0] != "/api/generate":
            return send_json(self, 404, {"ok": False, "error": "Not found"})
        try:
            corpus = get_corpus()
            details = corpus.describe()
            details.update({
                "assistant_ready": True,
                "assistant_model": "Neuron Forge · local assistant with 250M research architecture configured",
                "ai_mode": "offline-local",
                "external_ai_api": False,
                "conversation_memory": "browser-local; selected excerpts are sent only with chat requests",
                "model_target": {
                    "name": "Neuron Forge 250M",
                    "parameters": 248144832,
                    "status": "architecture-configured; training not started",
                    "trained": False,
                    "chat_enabled": False,
                    "training_device": "dedicated CUDA GPU required",
                    "corpus": "general-domain corpus not yet provided",
                },
            })
            send_json(self, 200, {"ok": True, **details})
        except Exception as exc:
            print(f"Neuron Forge health check failed: {type(exc).__name__}", flush=True)
            send_json(self, 503, {
                "ok": False,
                "assistant_ready": False,
                "error": "The local conversation engine or reference library could not be loaded.",
            })

    def do_POST(self):
        if self.path.split("?", 1)[0] != "/api/generate":
            return send_json(self, 404, {"error": "Not found"})
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            return send_json(self, 400, {"error": "Invalid content length"})
        if length < 1 or length > 60000:
            return send_json(self, 413, {"error": "Request must be between 1 byte and 60 KB"})

        try:
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise ValueError("JSON body must be an object")
            messages = normalize_messages(payload)
            memories = normalize_memory_entries(payload.get("memories"))
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            return send_json(self, 400, {"error": str(exc)})

        try:
            latest_user = next(item["content"] for item in reversed(messages) if item["role"] == "user")
            corpus = get_corpus()
            passages = corpus.search(latest_user, limit=2) if should_search_shakespeare(latest_user, corpus) else []
            answer = get_engine().respond(messages, passages, memories)
            send_json(self, 200, {
                "text": answer,
                "model": "neuron-forge-local",
                "model_label": "NEURON FORGE",
                "neural_experimental": False,
                "assistant": "Neuron Forge",
                "mode": "offline-local",
                "external_api_used": False,
            })
        except Exception as exc:
            print(f"Neuron Forge local request failed: {type(exc).__name__}", flush=True)
            send_json(self, 503, {
                "error": "The local assistant could not complete that response."
            })

    def log_message(self, format, *args):
        return

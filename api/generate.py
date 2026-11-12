"""Vercel Python Function for Neuron Forge's local Shakespeare corpus engine."""
from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler
from pathlib import Path
import sys
from threading import Lock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from neuronforge_engine.web_runtime import load_web_model  # noqa: E402

_model = None
_model_lock = Lock()

def get_model():
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                _model = load_web_model(ROOT / "public" / "shakespeare_index.json")
    return _model

def send_json(handler, status, payload):
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Cache-Control", "no-store")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.split("?", 1)[0] != "/api/generate":
            return send_json(self, 404, {"ok": False, "error": "Not found"})
        try:
            send_json(self, 200, {"ok": True, **get_model().describe()})
        except Exception as exc:
            send_json(self, 503, {"ok": False, "error": f"Corpus index unavailable: {exc}"})

    def do_POST(self):
        if self.path.split("?", 1)[0] != "/api/generate":
            return send_json(self, 404, {"error": "Not found"})
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            return send_json(self, 400, {"error": "Invalid content length"})
        if length < 1 or length > 12000:
            return send_json(self, 413, {"error": "Request must be between 1 byte and 12 KB"})
        try:
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise ValueError("JSON body must be an object")
            prompt = payload.get("prompt", "")
            if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 2000:
                raise ValueError("Prompt must contain 1–2,000 characters")
            max_results = int(payload.get("max_results", 2))
            if not 1 <= max_results <= 3:
                raise ValueError("max_results must be between 1 and 3")
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            return send_json(self, 400, {"error": str(exc)})
        try:
            answer = get_model().generate(prompt.strip(), max_results=max_results)
            send_json(self, 200, {"text": answer, "model": "Neuron Forge · Shakespeare Corpus"})
        except Exception as exc:
            send_json(self, 503, {"error": f"Corpus search failed: {exc}"})

    def log_message(self, format, *args):
        return

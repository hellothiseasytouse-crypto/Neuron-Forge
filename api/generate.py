"""Vercel Python Function for the standard-library Neuron Forge web demo."""
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
                _model = load_web_model(ROOT / "public" / "model.json")
    return _model

def send_json(handler, status, payload):
    body = json.dumps(payload).encode("utf-8")
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
            model = get_model()
            send_json(self, 200, {
                "ok": True, "model": "Neuron Forge Web RNN",
                "architecture": "character-level recurrent neural network",
                "parameters": model.parameter_count,
                "training_steps": model.training_steps,
                "training_chars": model.training_chars,
                "trained_from_scratch": True,
            })
        except Exception as exc:
            send_json(self, 503, {"ok": False, "error": f"Web model unavailable: {exc}"})

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
            max_new_tokens = int(payload.get("max_new_tokens", 96))
            temperature = float(payload.get("temperature", 0.8))
            top_k = int(payload.get("top_k", 5))
            if not 1 <= max_new_tokens <= 128:
                raise ValueError("max_new_tokens must be between 1 and 128")
            if not 0.15 <= temperature <= 1.5:
                raise ValueError("temperature must be between 0.15 and 1.5")
            if not 1 <= top_k <= 128:
                raise ValueError("top_k must be between 1 and 128")
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            return send_json(self, 400, {"error": str(exc)})
        try:
            model = get_model()
            seed = "User: " + prompt.strip() + "\nAssistant:"
            answer = model.generate(seed, max_new_tokens, temperature, top_k)
            send_json(self, 200, {"text": answer, "model": "Neuron Forge Web RNN"})
        except Exception as exc:
            send_json(self, 503, {"error": f"Hosted generation failed: {exc}"})

    def log_message(self, format, *args):
        return

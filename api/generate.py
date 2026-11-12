"""Vercel Python function for the hosted Neuron Forge demo model."""
import json
import sys
from pathlib import Path
from threading import Lock
from http.server import BaseHTTPRequestHandler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

_runtime = None
_runtime_lock = Lock()


def get_runtime():
    global _runtime
    if _runtime is None:
        with _runtime_lock:
            if _runtime is None:
                from neuronforge_engine.checkpoint import load_model
                checkpoint_path = ROOT / "checkpoints" / "latest.pt"
                _runtime = load_model(checkpoint_path, device="cpu")
    return _runtime


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
            model, tokenizer, checkpoint = get_runtime()
            send_json(self, 200, {
                "ok": True,
                "model": "Neuron Forge 0.2",
                "parameters": model.num_parameters(),
                "training_steps": int(checkpoint.get("step", 0)),
                "trained_from_scratch": True,
            })
        except Exception:
            send_json(self, 503, {
                "ok": False,
                "error": "The model checkpoint is not available yet. Check the deployment build logs.",
            })

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
            top_k = int(payload.get("top_k", 40))
            top_p = float(payload.get("top_p", 0.95))
            if not 1 <= max_new_tokens <= 128:
                raise ValueError("max_new_tokens must be between 1 and 128")
            if not 0 <= temperature <= 2:
                raise ValueError("temperature must be between 0 and 2")
            if not 1 <= top_k <= 320:
                raise ValueError("top_k must be between 1 and 320")
            if not 0 < top_p <= 1:
                raise ValueError("top_p must be greater than 0 and at most 1")
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            return send_json(self, 400, {"error": str(exc)})

        try:
            from neuronforge_engine.generation import generate_from_model
            model, tokenizer, _ = get_runtime()
            model_prompt = "User: " + prompt.strip() + "\nAssistant:"
            generated = generate_from_model(
                model, tokenizer, model_prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=1.05,
                device="cpu",
            )
            answer = generated[len(model_prompt):].strip() if generated.startswith(model_prompt) else generated.strip()
            send_json(self, 200, {"text": answer, "model": "Neuron Forge 0.2"})
        except Exception:
            send_json(self, 503, {
                "error": "Generation failed in the hosted model runtime. Please try a shorter prompt.",
            })

    def log_message(self, format, *args):
        return

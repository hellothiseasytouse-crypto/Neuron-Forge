"""Optional local-only FastAPI interface."""
from pathlib import Path
from threading import Lock

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .checkpoint import load_model
from .generation import generate_from_model


class GenerateRequest(BaseModel):
    prompt: str = Field(default="", max_length=20_000)
    max_new_tokens: int = Field(default=160, ge=1, le=1024)
    temperature: float = Field(default=0.8, ge=0.0, le=3.0)
    top_k: int | None = Field(default=40, ge=1, le=100_000)
    top_p: float = Field(default=0.95, gt=0.0, le=1.0)
    repetition_penalty: float = Field(default=1.05, gt=0.0, le=5.0)


def create_app(checkpoint_path: str, device: str = "cpu") -> FastAPI:
    model, tokenizer, _ = load_model(checkpoint_path, device=device)
    lock = Lock()
    app = FastAPI(title="Neuron Forge", version="0.2.0",
                  description="Local text generation from a model trained from scratch.")

    @app.get("/health")
    def health():
        return {"status": "ok", "model": "Neuron Forge", "parameters": model.num_parameters()}

    @app.get("/")
    def home():
        page = Path(__file__).parent / "static" / "index.html"
        return FileResponse(page)

    @app.post("/generate")
    def generate_route(request: GenerateRequest):
        try:
            with lock:
                text = generate_from_model(
                    model, tokenizer, request.prompt, request.max_new_tokens,
                    request.temperature, request.top_k, request.top_p,
                    request.repetition_penalty, device,
                )
            return {"text": text}
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return app

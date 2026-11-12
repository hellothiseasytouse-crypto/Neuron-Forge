"""Text generation / sampling."""


def generate(checkpoint: str, prompt: str, max_new_tokens: int = 200,
             temperature: float = 1.0, top_k: int | None = None) -> str:
    # TODO: load checkpoint, encode prompt, sample token by token, decode
    raise NotImplementedError

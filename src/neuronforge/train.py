"""Training loop."""
from .config import Config


def train(cfg: Config) -> None:
    # TODO: load text -> tokenizer -> dataset -> GPT -> AdamW loop
    #       periodic eval loss, save checkpoint (model + tokenizer + config)
    raise NotImplementedError

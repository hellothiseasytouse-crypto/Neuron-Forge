class CharTokenizer:
    """Character-level tokenizer. TODO: replace/extend with BPE."""

    def __init__(self, chars: list[str]):
        self.itos = {i: c for i, c in enumerate(chars)}
        self.stoi = {c: i for i, c in self.itos.items()}

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        return cls(sorted(set(text)))

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def encode(self, text: str) -> list[int]:
        return [self.stoi[c] for c in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids)

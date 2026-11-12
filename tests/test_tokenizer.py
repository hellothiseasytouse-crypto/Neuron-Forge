from neuronforge.tokenizer import CharTokenizer


def test_roundtrip():
    tok = CharTokenizer.from_text("hello world")
    assert tok.decode(tok.encode("hello")) == "hello"

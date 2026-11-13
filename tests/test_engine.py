import torch

from neuronforge_engine.config import ModelConfig
from neuronforge_engine.data import encode_and_split, get_batch
from neuronforge_engine.generation import generate_from_model
from neuronforge_engine.model import GPT
from neuronforge_engine.tokenizer import ByteBPETokenizer


def test_bpe_roundtrip_utf8_and_state():
    text = "hello hello — Neuron Forge 🤖\n" * 10
    tokenizer = ByteBPETokenizer.train(text, vocab_size=300)
    assert tokenizer.decode(tokenizer.encode(text)) == text
    restored = ByteBPETokenizer.from_state_dict(tokenizer.to_state_dict())
    assert restored.decode(restored.encode(text)) == text


def test_batch_shapes_and_shift():
    data = torch.arange(50)
    x, y = get_batch(data, block_size=8, batch_size=4)
    assert x.shape == (4, 8)
    assert y.shape == (4, 8)
    assert torch.equal(x[:, 1:], y[:, :-1])


def test_model_forward_loss_and_generation():
    text = "hello world " * 60
    tokenizer = ByteBPETokenizer.train(text, vocab_size=270)
    cfg = ModelConfig(vocab_size=tokenizer.vocab_size, block_size=32, n_layer=2, n_head=2, n_embd=32, dropout=0.0)
    model = GPT(cfg)
    tokens = torch.tensor([list(range(16))], dtype=torch.long)
    logits, loss = model(tokens, tokens)
    assert logits.shape == (1, 16, tokenizer.vocab_size)
    assert loss is not None and torch.isfinite(loss)
    output = generate_from_model(model, tokenizer, "hi", max_new_tokens=3, temperature=0.0)
    assert output.startswith("hi")


def test_split_returns_train_and_validation():
    text = "abcde " * 100
    tokenizer = ByteBPETokenizer()
    train, val = encode_and_split(text, tokenizer)
    assert train.ndim == val.ndim == 1
    assert len(train) > len(val) > 0



def test_transformer_attention_is_causal():
    torch.manual_seed(7)
    cfg = ModelConfig(vocab_size=300, block_size=16, n_layer=2, n_head=2, n_embd=32, dropout=0.0)
    model = GPT(cfg).eval()
    first = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    changed_future = torch.tensor([[1, 2, 3, 99]], dtype=torch.long)
    with torch.no_grad():
        logits_a, _ = model(first)
        logits_b, _ = model(changed_future)
    assert torch.allclose(logits_a[:, :3], logits_b[:, :3], atol=1e-6, rtol=1e-5)


def test_transformer_uses_rope_and_swiglu_blocks():
    cfg = ModelConfig(vocab_size=300, block_size=16, n_layer=1, n_head=2, n_embd=32, dropout=0.0)
    model = GPT(cfg)
    assert model.blocks[0].mlp.__class__.__name__ == "SwiGLU"
    assert model.blocks[0].attn.__class__.__name__ == "CausalSelfAttention"
    assert model.num_parameters() > 0

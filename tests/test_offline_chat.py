from pathlib import Path

from neuronforge_engine.offline_chat import LocalConversationEngine, _safe_math


def make_engine(tmp_path: Path) -> LocalConversationEngine:
    # The source path is accepted for compatibility, but is no longer loaded
    # into or used by the general assistant.
    source = tmp_path / "legacy-source.txt"
    source.write_text("Legacy project notes.", encoding="utf-8")
    return LocalConversationEngine(source)


def test_greeting_is_natural(tmp_path):
    assert make_engine(tmp_path).respond([{"role": "user", "content": "yo"}]).startswith("Hey!")


def test_identity_is_neuron_forge_not_fai(tmp_path):
    answer = make_engine(tmp_path).respond([{"role": "user", "content": "Who are you?"}])
    assert "Neuron Forge" in answer
    assert "FAI" not in answer


def test_unreliable_neural_checkpoint_is_not_used_as_chat_output(tmp_path):
    answer = make_engine(tmp_path).respond([{"role": "user", "content": "Explain quantum computing."}])
    assert "reliable local knowledge" in answer
    assert "FAI" not in answer
    assert len(answer) > 80


def test_safe_arithmetic():
    assert _safe_math("12 * (3 + 2)") == "60"
    assert _safe_math("10 / 4") == "2.5"
    assert _safe_math("__import__('os').system('whoami')") is None
    assert _safe_math("2 ** 100") is None


def test_nested_arithmetic_is_answered(tmp_path):
    answer = make_engine(tmp_path).respond([{"role": "user", "content": "what is 12 * (3 + 2)?"}])
    assert answer == "12 * (3 + 2) = 60"


def test_chat_history_remembers_name(tmp_path):
    messages = [
        {"role": "user", "content": "My name is Alex"},
        {"role": "assistant", "content": "Got it, Alex."},
        {"role": "user", "content": "Do you remember my name?"},
    ]
    assert "Alex" in make_engine(tmp_path).respond(messages)


def test_cross_chat_memory_recalls_previous_topic(tmp_path):
    messages = [{"role": "user", "content": "Continue where we left off"}]
    memories = [
        {"role": "user", "content": "I am building a project called Eclipse."},
        {"role": "assistant", "content": "We were planning its story and gameplay."},
    ]
    answer = make_engine(tmp_path).respond(messages, memories=memories)
    assert "Eclipse" in answer
    assert "context" in answer.lower()


def test_shakespeare_passage_uses_real_indexed_text(tmp_path):
    answer = make_engine(tmp_path).respond(
        [{"role": "user", "content": "Find a passage"}],
        passages=[{"work": "Macbeth", "text": "Fair is foul, and foul is fair."}],
    )
    assert "Macbeth" in answer
    assert "Fair is foul" in answer


def test_thanks_is_friendly(tmp_path):
    answer = make_engine(tmp_path).respond([{"role": "user", "content": "Thanks"}])
    assert answer.startswith("You're welcome.")

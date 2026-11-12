from pathlib import Path

from neuronforge_engine.offline_chat import LocalConversationEngine, _safe_math


def make_engine(tmp_path: Path) -> LocalConversationEngine:
    source = tmp_path / "FAI_SOURCE.txt"
    source.write_text(
        "THE FOUR SYSTEMS\nPERSONALITY-AI\nWhat the subject is like.\n"
        "FEELINGS-AI\nHow the subject reacts emotionally.\n"
        "RULE-AI\nThe behavioral rules discovered throughout the archive.\n"
        "TRAITS-AI\nThe smaller characteristics that make the subject recognizable.\n",
        encoding="utf-8",
    )
    return LocalConversationEngine(source)


def test_greeting_is_local_and_natural(tmp_path):
    engine = make_engine(tmp_path)
    assert engine.respond([{"role": "user", "content": "yo"}]).startswith("Hey.")


def test_fai_four_systems_are_returned_without_rewriting_terms(tmp_path):
    engine = make_engine(tmp_path)
    answer = engine.respond([{"role": "user", "content": "What are the four systems of FAI?"}])
    assert "PERSONALITY-AI — What the subject is like." in answer
    assert "FEELINGS-AI — How the subject reacts emotionally." in answer
    assert "RULE-AI — The behavioral rules discovered throughout the archive." in answer
    assert "TRAITS-AI — The smaller characteristics that make the subject recognizable." in answer
    assert "FAI-AI" in answer


def test_missing_subject_archive_is_not_fabricated(tmp_path):
    engine = make_engine(tmp_path)
    answer = engine.respond([{"role": "user", "content": "Reconstruct the FAI subject"}])
    assert "does not contain the underlying months" in answer
    assert "invented memories" in answer


def test_profanity_is_not_blocked(tmp_path):
    engine = make_engine(tmp_path)
    answer = engine.respond([{"role": "user", "content": "This is fucking annoying"}])
    assert answer


def test_safe_arithmetic():
    assert _safe_math("12 * (3 + 2)") == "60"
    assert _safe_math("10 / 4") == "2.5"
    assert _safe_math("__import__('os').system('whoami')") is None
    assert _safe_math("2 ** 100") is None


def test_chat_history_remembers_name_within_conversation(tmp_path):
    engine = make_engine(tmp_path)
    messages = [
        {"role": "user", "content": "My name is Alex"},
        {"role": "assistant", "content": "Got it, Alex."},
        {"role": "user", "content": "Do you remember my name?"},
    ]
    assert "Alex" in engine.respond(messages)

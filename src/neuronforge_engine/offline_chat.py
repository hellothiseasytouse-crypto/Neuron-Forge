"""Offline conversation runtime built around the Dying Ember source of truth.

This is a deterministic, inspectable starter runtime, not a trained foundation
model. It deliberately refuses to invent missing FAI evidence.
"""
from __future__ import annotations

import ast
import operator
import re
from pathlib import Path
from typing import Any

_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_math(expression: str) -> str | None:
    """Evaluate a deliberately small arithmetic grammar; never use eval()."""
    expression = expression.replace(",", "")
    if len(expression) > 100 or not re.fullmatch(r"[\d\s.+*/()%\-]+", expression):
        return None

    def visit(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in _OPERATORS:
            left, right = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Pow) and (abs(right) > 8 or abs(left) > 1_000_000):
                raise ValueError("exponent outside the supported range")
            return float(_OPERATORS[type(node.op)](left, right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPERATORS:
            return float(_OPERATORS[type(node.op)](visit(node.operand)))
        raise ValueError("unsupported expression")

    try:
        result = visit(ast.parse(expression, mode="eval"))
        if not (float("-inf") < result < float("inf")) or abs(result) > 1e12:
            return None
        if result.is_integer():
            return str(int(result))
        return f"{result:.10g}"
    except (ArithmeticError, SyntaxError, TypeError, ValueError, OverflowError):
        return None


class ChatTone:
    """Controls the website assistant's voice; it is not FAI's personality model."""

    def respond(self, message: str, mood: str) -> str:
        # Natural language and profanity are allowed; no profanity filter is applied.
        return "casual" if mood in {"frustrated", "excited"} or len(message.split()) < 12 else "clear"


class ConversationSignals:
    """Detects obvious cues in the current chat; this is not FAI's feelings model."""

    _cues = {
        "frustrated": ("angry", "annoyed", "frustrated", "fuck", "fucking", "shit", "damn"),
        "excited": ("awesome", "great", "love this", "lets go", "let's go", "amazing"),
        "sad": ("sad", "upset", "lonely", "depressed", "hurt"),
        "uncertain": ("confused", "not sure", "unsure", "maybe", "i don't know"),
    }

    def detect(self, message: str) -> str:
        lowered = message.lower()
        for mood, terms in self._cues.items():
            if any(term in lowered for term in terms):
                return mood
        return "neutral"


class EvidencePolicy:
    """Applies the source document's evidence-versus-assumption warning to answers."""

    def __init__(self, source_text: str):
        self.source_text = source_text

    def evidence_answer(self) -> str:
        return (
            "The Dying Ember source says the archive is incomplete and that "
            "nothing is to be treated as absolute without evidence. It defines "
            "four systems exactly as follows:\n\n"
            "PERSONALITY-AI — What the subject is like.\n"
            "FEELINGS-AI — How the subject reacts emotionally.\n"
            "RULE-AI — The behavioral rules discovered throughout the archive.\n"
            "TRAITS-AI — The smaller characteristics that make the subject recognizable.\n\n"
            "Together, these systems will form the final reconstruction: FAI-AI."
        )

    def missing_evidence_answer(self) -> str:
        return (
            "The source document defines the mission and the four systems, but it "
            "does not contain the underlying months of subject conversations. "
            "That means there is not enough subject evidence here to build a "
            "reliable reconstruction yet. I will not fill those gaps with invented "
            "memories, traits, or emotional claims."
        )


# Definitions copied from the original Dying Ember source document.
# These are specifications, not claims that the subject has been reconstructed.
FAI_SYSTEMS = (
    ("PERSONALITY-AI", "What the subject is like."),
    ("FEELINGS-AI", "How the subject reacts emotionally."),
    ("RULE-AI", "The behavioral rules discovered throughout the archive."),
    ("TRAITS-AI", "The smaller characteristics that make the subject recognizable."),
)


class LocalConversationEngine:
    """A small local chat prototype; it makes no remote model or API calls."""

    def __init__(self, source_path: str | Path):
        path = Path(source_path)
        self.source_text = path.read_text(encoding="utf-8") if path.is_file() else ""
        self.chat_tone = ChatTone()
        self.conversation_signals = ConversationSignals()
        self.evidence_policy = EvidencePolicy(self.source_text)

    @staticmethod
    def _last_user(messages: list[dict[str, str]]) -> str:
        return next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")

    @staticmethod
    def _name_from_history(messages: list[dict[str, str]]) -> str | None:
        for item in reversed(messages):
            if item["role"] != "user":
                continue
            match = re.search(r"\b(?:my name is|call me)\s+([A-Za-z][A-Za-z0-9_-]{0,30})", item["content"], re.I)
            if match:
                return match.group(1)
        return None

    def respond(self, messages: list[dict[str, str]], passages: list[dict[str, Any]] | None = None) -> str:
        message = self._last_user(messages).strip()
        lowered = message.lower().strip()
        mood = self.conversation_signals.detect(message)
        tone = self.chat_tone.respond(message, mood)

        if not message:
            return "Send me a message and we can start."

        if re.search(r"\bwho is fai\b|\bsubject identity\b", lowered):
            return (
                "The Dying Ember source labels FAI as the subject, but lists the "
                "objective as unknown. It does not provide the subject's identity "
                "or enough conversation evidence to describe who they are. It says "
                "the reconstruction must be built from evidence, not assumption."
            )

        if re.search(r"\b(?:what is|who is|tell me about|explain)\b.*\b(?:fai|dying ember)\b|\bfai-ai\b|\bthe four systems\b", lowered):
            return self.evidence_policy.evidence_answer()

        if re.search(r"\b(?:reconstruct|reconstruction|simulate|copy|personality)\b", lowered) and re.search(r"\b(?:fai|subject|archive)\b", lowered):
            return self.evidence_policy.missing_evidence_answer()

        if re.fullmatch(r"(?:hi|hello|hey|yo|sup|heya|hii+)[!. ]*", lowered):
            return "Hey. I'm here. What's up?"

        if re.search(r"\b(?:how are you|how's it going|hows it going)\b", lowered):
            return "I'm running locally and ready to chat. What are you working on?"

        if re.search(r"\b(?:who are you|what are you|your name)\b", lowered):
            return (
                "I'm Neuron Forge. This version runs an offline conversation engine "
                "with the Dying Ember source document and the local Shakespeare library. "
                "It does not call an external AI API, and it is not yet a trained general-purpose model."
            )

        if re.search(r"\b(?:what can you do|help|capabilities)\b", lowered):
            return (
                "Right now I can hold a basic conversation, remember details within this chat, "
                "do simple arithmetic, answer questions about the exact Dying Ember source, "
                "and search the local Shakespeare collection. I cannot reliably answer arbitrary "
                "world-knowledge questions yet; the from-scratch language model still needs training."
            )

        if re.search(r"\b(?:what did i say|what was my name|do you remember my name)\b", lowered):
            name = self._name_from_history(messages[:-1])
            if name:
                return f"You told me to call you {name} earlier in this conversation."
            earlier = [m["content"] for m in messages[:-1] if m["role"] == "user"]
            if earlier:
                return "Earlier in this chat, you said: " + earlier[-1][:350]
            return "I don't have an earlier message to refer to in this conversation yet."

        if re.match(r"^(?:my name is|call me)\s+", lowered):
            name = self._name_from_history(messages)
            return f"Got it, {name}. I'll use that name during this conversation."

        if re.search(r"\b(?:thanks|thank you|thx)\b", lowered):
            return "You're welcome. What else?"

        if re.search(r"\b(?:fuck you|you suck|you're useless|you are useless)\b", lowered):
            return "Fair enough — tell me what failed and I'll try to fix the actual problem."

        if re.search(r"\b(?:joke|make me laugh)\b", lowered):
            return "Why did the programmer quit their job? They didn't get arrays. 🙃"

        math_match = re.fullmatch(
            r"\s*(?:(?:what is|calculate|solve|compute)\s+)?([0-9().+*/%\- ]+)\s*[?!.]*\s*",
            lowered,
        )
        if math_match:
            expression = math_match.group(1).strip()
            result = _safe_math(expression)
            if result is not None and re.search(r"[+*/%]", expression):
                return f"{expression} = {result}"

        if re.search(r"\b(?:fai|dying ember|subject|archive|personality-ai|feelings-ai|rule-ai|traits-ai)\b", lowered):
            return self.evidence_policy.evidence_answer() + "\n\n" + self.evidence_policy.missing_evidence_answer()

        if passages:
            first = passages[0]
            work = first.get("work", "the Shakespeare collection")
            excerpt = " ".join(str(first.get("text", "")).split())[:650]
            return (
                f"I found a relevant passage in {work} in the local library:\n\n"
                f"“{excerpt}”\n\n"
                "This is a source excerpt, not a complete interpretation. I can search "
                "the collection, but this offline prototype cannot yet explain every topic."
            )

        if mood == "sad":
            return "That sounds rough. Want to tell me what happened? I'll listen."
        if mood == "frustrated":
            return "I hear you. Tell me what you need fixed, and I'll focus on that instead of giving you a lecture."
        if mood == "excited":
            return "Let's go. Tell me more."
        if tone == "clear":
            return (
                "I can follow what you're saying, but I don't have a trained general-purpose "
                "language model behind this offline prototype yet. I won't make up an answer. "
                "I can still help with basic conversation, simple calculations, the Dying Ember "
                "source document, and the local Shakespeare collection."
            )
        return (
            "I'm here. I can chat, keep track of what you've said in this conversation, "
            "and work with the local material. I don't know enough to answer that one reliably yet."
        )

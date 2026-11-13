"""Conversation layer for Neuron Forge's local assistant prototype.

This module deliberately avoids presenting raw, untrained language-model
completions as answers. It provides memory, arithmetic, and reference-library
routing while the from-scratch model remains a research checkpoint.
"""
from __future__ import annotations

import ast
import operator
import re
from pathlib import Path
from typing import Any

from .memory import select_relevant_memories

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
        return str(int(result)) if result.is_integer() else f"{result:.10g}"
    except (ArithmeticError, SyntaxError, TypeError, ValueError, OverflowError):
        return None


class LocalConversationEngine:
    """A transparent local assistant layer; it does not impersonate a person."""

    def __init__(self, source_path: str | Path | None = None):
        # source_path remains accepted for compatibility with older local callers.
        self.source_path = Path(source_path) if source_path else None

    @staticmethod
    def _last_user(messages: list[dict[str, str]]) -> str:
        return next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")

    @staticmethod
    def _name_from_history(
        messages: list[dict[str, str]],
        memories: list[dict[str, str]] | None = None,
    ) -> str | None:
        for item in reversed(list(memories or []) + list(messages)):
            if item.get("role") != "user":
                continue
            match = re.search(
                r"\b(?:my name is|call me)\s+([A-Za-z][A-Za-z0-9_-]{0,30})",
                item.get("content", ""),
                re.I,
            )
            if match:
                return match.group(1)
        return None

    def respond(
        self,
        messages: list[dict[str, str]],
        passages: list[dict[str, Any]] | None = None,
        memories: list[dict[str, str]] | None = None,
    ) -> str:
        message = self._last_user(messages).strip()
        lowered = message.lower().strip()
        memories = memories or []

        if not message:
            return "Send me a message and we can get started."

        if re.search(
            r"\b(?:what did i say|what did we discuss|what were we talking about|"
            r"where did we leave off|continue where we left off|what do you remember about me)\b",
            lowered,
        ):
            name = self._name_from_history(messages[:-1], memories)
            if name and re.search(r"\b(?:my name|name is)\b", lowered):
                return f"You told me your name is {name}. I found that in the conversation history available to this browser."
            records = memories + messages[:-1]
            recalled = select_relevant_memories(message, records, limit=6)
            if recalled:
                excerpts = [
                    f"{item['role'].title()}: {item['content'][:360]}"
                    for item in recalled[-5:]
                ]
                return "Here's the earlier context I could retrieve:\n\n" + "\n".join(excerpts)
            return "I couldn't find matching earlier context in the conversations available to this browser."

        if len(message.split()) <= 10 and re.search(
            r"\b(?:continue|do it|make it better|make it stronger|that project|the project|"
            r"same thing|go on|keep going)\b",
            lowered,
        ):
            context = select_relevant_memories(message, memories + messages[:-1], limit=4)
            if context:
                excerpts = [
                    f"{item['role'].title()}: {item['content'][:300]}"
                    for item in context[-4:]
                ]
                return (
                    "I found the previous context:\n\n"
                    + "\n".join(excerpts)
                    + "\n\nI can use that context as we continue."
                )
            return "I can continue, but I can't see enough of the earlier context. Remind me what we were working on."

        if re.fullmatch(r"(?:hi|hello|hey|yo|sup|heya|hii+)[!. ]*", lowered):
            return "Hey! What are you working on?"

        if re.search(r"\b(?:how are you|how's it going|hows it going)\b", lowered):
            return "I'm running locally and ready to help. What's on your mind?"

        if re.search(r"\b(?:who are you|what are you|your name)\b", lowered):
            return (
                "I'm Neuron Forge, a from-scratch AI research project with a local chat layer. "
                "I can use the context in this conversation, recall available browser-saved chats, "
                "do constrained calculations, and search the included reference library. "
                "The 1M-parameter neural checkpoint is still experimental, so I don't pass its "
                "unfiltered text output off as reliable conversation."
            )

        if re.search(r"\b(?:what can you do|your capabilities|capabilities|help)\b", lowered):
            return (
                "I can keep track of this conversation, retrieve relevant context from chats saved "
                "in this browser, solve supported arithmetic, and search the local Shakespeare "
                "reference collection. I can also help organize a problem when you provide its "
                "details. Broad factual answering and open-ended reasoning are not reliable yet: "
                "the from-scratch neural checkpoint is a small research model, not an instruction-trained assistant."
            )

        if re.search(r"\b(?:what did i say|what was my name|do you remember my name)\b", lowered):
            name = self._name_from_history(messages[:-1], memories)
            if name:
                return f"You told me your name is {name}. That detail is in the conversation context available to me."
            earlier = [m["content"] for m in messages[:-1] if m["role"] == "user"]
            if earlier:
                return "The most recent thing you told me was: " + earlier[-1][:350]
            return "I don't have an earlier message to refer to yet."

        if re.match(r"^(?:my name is|call me)\s+", lowered):
            name = self._name_from_history(messages, memories)
            return f"Got it. I'll use {name} in this conversation." if name else "Got it."

        if re.search(r"\b(?:thanks|thank you|thx)\b", lowered):
            return "You're welcome. What should we tackle next?"

        math_match = re.fullmatch(
            r"\s*(?:(?:what is|calculate|solve|compute)\s+)?([0-9().+*/%\- ]+)\s*[?!.]*\s*",
            lowered,
        )
        if math_match:
            expression = math_match.group(1).strip()
            result = _safe_math(expression)
            if result is not None and re.search(r"[+*/%]", expression):
                return f"{expression} = {result}"

        if passages:
            first = passages[0]
            work = first.get("work", "the local reference collection")
            excerpt = " ".join(str(first.get("text", "")).split())[:650]
            return (
                f"I found this passage in {work}:\n\n“{excerpt}”\n\n"
                "That's the source text. I can help examine it, but I won't invent lines that aren't in the collection."
            )

        if re.search(r"\b(?:fuck you|you suck|you're useless|you are useless)\b", lowered):
            return "Fair criticism. Tell me what failed and I'll focus on fixing that."

        if re.search(r"\b(?:write|draft|make|build|create|plan|design)\b", lowered):
            return (
                "I can help structure that. Send the goal, what you already have, and any constraints "
                "that matter. This local build doesn't yet have a reliable instruction-trained model, "
                "so I'll be clear about what I can verify instead of inventing a finished result."
            )

        return (
            "I follow what you're asking. I don't have enough reliable local knowledge to answer that "
            "confidently yet, and the small research checkpoint isn't dependable enough to fill the gap. "
            "Share the relevant details or a source, and I can work through the information with you."
        )

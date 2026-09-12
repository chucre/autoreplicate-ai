"""Common interface every LLM decision backend implements.

Agent talks to whichever connector is configured through this one method —
it doesn't know or care whether the backend is a subprocess call to the
`claude` CLI, an HTTP call to OpenRouter, Abacus, or anything else. Adding a
new provider means adding one module in this package and one entry in the
registry in `core/llm_connectors/__init__.py` — nothing in core/agent.py
has to change.
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class LlmConnector(ABC):
    """Turns a (system prompt, user prompt) pair into the model's raw text reply."""

    @abstractmethod
    def complete(self, *, system_prompt: str, user_prompt: str, temperature: float) -> str:
        """Return the model's plain-text response.

        Raise on any backend failure — callers don't retry or fall back
        automatically.
        """

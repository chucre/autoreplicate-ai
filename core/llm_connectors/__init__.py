"""Registry of available LLM decision backends.

To add a new provider (OpenRouter, Abacus, ...): implement `LlmConnector`
(see `base.py`) in a new module in this package, then register a factory
for it in `_CONNECTORS` below. `Agent` only ever depends on `get_connector`
and the `LlmConnector` interface, so nothing outside this package needs to
change.
"""
from __future__ import annotations

import os
from typing import Callable, Dict, Optional

from core.llm_connectors.base import LlmConnector
from core.llm_connectors.claude_cli import ClaudeCliConnector

_CONNECTORS: Dict[str, Callable[[], LlmConnector]] = {
    "claude_cli": ClaudeCliConnector,
}


def get_connector(name: Optional[str] = None) -> LlmConnector:
    """Build the named connector, or AGENT_LLM_CONNECTOR from the
    environment, defaulting to "claude_cli" if neither is set."""
    name = name or os.environ.get("AGENT_LLM_CONNECTOR", "claude_cli")
    try:
        factory = _CONNECTORS[name]
    except KeyError:
        raise ValueError(f"unknown LLM connector {name!r}; available: {sorted(_CONNECTORS)}") from None
    return factory()

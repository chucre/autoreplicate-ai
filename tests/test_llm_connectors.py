import pytest

from core.llm_connectors import get_connector
from core.llm_connectors.claude_cli import ClaudeCliConnector


def test_default_connector_is_claude_cli():
    assert isinstance(get_connector(), ClaudeCliConnector)


def test_get_connector_by_explicit_name():
    assert isinstance(get_connector("claude_cli"), ClaudeCliConnector)


def test_unknown_connector_name_raises():
    with pytest.raises(ValueError):
        get_connector("does_not_exist")


def test_env_var_selects_connector(monkeypatch):
    monkeypatch.setenv("AGENT_LLM_CONNECTOR", "claude_cli")
    assert isinstance(get_connector(), ClaudeCliConnector)

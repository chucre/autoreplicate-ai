"""ClaudeCliConnector: talks to Claude through the `claude` CLI in headless
mode, authenticated by a Claude subscription's CLAUDE_CODE_OAUTH_TOKEN (from
`claude setup-token`) instead of a metered Anthropic API key.

Each call spins up a full Claude Code session — real latency, session
overhead — and draws on the subscription's usage limits rather than being
billed per token. Fine for a slow simulation loop, not a tight one.
"""
from __future__ import annotations

import json
import os
import subprocess
from typing import Optional

from core.llm_connectors.base import LlmConnector


class ClaudeCliConnector(LlmConnector):
    def __init__(self, model: Optional[str] = None) -> None:
        self.model = model or os.environ.get("AGENT_MODEL")

    def complete(self, *, system_prompt: str, user_prompt: str, temperature: float) -> str:
        # temperature isn't passed through - the CLI doesn't expose sampling params.
        if not os.environ.get("CLAUDE_CODE_OAUTH_TOKEN"):
            raise RuntimeError(
                "CLAUDE_CODE_OAUTH_TOKEN is not set. Generate one with `claude setup-token` "
                "(requires a Claude subscription) and export it before running the simulation."
            )

        command = ["claude", "-p", user_prompt, "--append-system-prompt", system_prompt, "--output-format", "json"]
        if self.model:
            command += ["--model", self.model]

        try:
            completed = subprocess.run(command, capture_output=True, text=True, timeout=120)
        except FileNotFoundError as exc:
            raise RuntimeError("the `claude` CLI was not found on PATH; install Claude Code to use it here") from exc

        if completed.returncode != 0:
            raise RuntimeError(f"claude CLI failed (exit {completed.returncode}): {completed.stderr.strip()}")

        payload = json.loads(completed.stdout)
        if payload.get("is_error"):
            raise RuntimeError(f"claude CLI returned an error: {payload}")
        return payload["result"].strip()

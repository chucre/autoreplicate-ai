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
            completed = subprocess.run(command, capture_output=True, text=True, timeout=180)
        except FileNotFoundError as exc:
            raise RuntimeError("the `claude` CLI was not found on PATH; install Claude Code to use it here") from exc
        except subprocess.TimeoutExpired as exc:
            # An invalid/expired CLAUDE_CODE_OAUTH_TOKEN makes the CLI retry
            # for a couple of minutes before it gives up, so this usually
            # means bad auth rather than a hung process.
            raise RuntimeError(
                "claude CLI timed out; if CLAUDE_CODE_OAUTH_TOKEN is stale or invalid, "
                "the CLI can spend ~2 minutes retrying before failing — regenerate the "
                "token with `claude setup-token`"
            ) from exc

        # `claude --output-format json` still writes a structured result to
        # stdout on failure (auth errors, bad model, refusals, ...) even
        # with a non-zero exit code — the informative message lives in
        # payload["result"], not in stderr, so parse stdout before giving up.
        payload = None
        if completed.stdout.strip():
            try:
                payload = json.loads(completed.stdout)
            except json.JSONDecodeError:
                payload = None

        if payload is not None:
            if payload.get("is_error"):
                status = payload.get("api_error_status")
                where = f" (HTTP {status})" if status else ""
                raise RuntimeError(f"claude CLI reported an error{where}: {payload.get('result', payload)}")
            return payload["result"].strip()

        raise RuntimeError(
            f"claude CLI failed (exit {completed.returncode}) with no parseable output: "
            f"stderr={completed.stderr.strip()!r} stdout={completed.stdout.strip()!r}"
        )

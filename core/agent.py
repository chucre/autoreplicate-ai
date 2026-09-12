"""Agent: an autonomous economic actor with a wallet, an action policy, and
a genome that shapes how it decides what to do next.

Each decision cycle asks an LLM to pick one of the agent's enabled tools,
checks that tool's category against ActionPolicy, debits its metabolic
cost from the wallet, and only then runs it. Every capability an agent can
possibly exercise bottoms out in a ToolSpec from `tools/`, and every
ToolSpec's category must already be allowlisted — the agent itself never
grants itself a new capability.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from decimal import Decimal
from typing import Callable, Dict, List, Optional, Tuple

from core.action_policy import ActionCategory, ActionPolicy
from core.wallet import Wallet

logger = logging.getLogger(__name__)

# Unset by default: let the `claude` CLI use whatever model it's configured
# for. Set AGENT_MODEL to pin a specific one (passed as `claude ... --model`).
DEFAULT_MODEL = os.environ.get("AGENT_MODEL")


@dataclass
class Genome:
    """Heritable parameters that make one agent's behavior differ from
    another's. Spawner mutates these when creating a child."""

    temperature: float = 0.7
    strategy_prompt: str = "Maximize wallet balance over time while staying within your allowed actions."
    enabled_tools: Tuple[str, ...] = ()


@dataclass
class ToolSpec:
    """One concrete capability an agent can invoke."""

    name: str
    category: ActionCategory
    cost: Decimal
    run: Callable[[], str]
    description: str = ""


LlmDecide = Callable[[str, List[ToolSpec], float], str]


class Agent:
    def __init__(
        self,
        name: str,
        wallet: Wallet,
        action_policy: ActionPolicy,
        genome: Genome,
        tools: Dict[str, ToolSpec],
        llm_decide: Optional[LlmDecide] = None,
        model: Optional[str] = DEFAULT_MODEL,
    ) -> None:
        self.name = name
        self.wallet = wallet
        self.action_policy = action_policy
        self.genome = genome
        self.tools = tools
        self.model = model
        self._llm_decide = llm_decide or self._default_llm_decide
        self.generation = 0
        self.alive = True

    def available_tools(self) -> List[ToolSpec]:
        """Tools this agent is allowed to use, that it can currently afford."""
        candidates = (self.tools[name] for name in self.genome.enabled_tools if name in self.tools)
        return [
            tool
            for tool in candidates
            if self.action_policy.is_allowed(tool.category) and tool.cost <= self.wallet.balance
        ]

    def step(self) -> Optional[str]:
        """Run one decision cycle: choose a tool, pay its metabolic cost, run it.

        Returns the tool's output, or None if the agent is extinct or has
        nothing it can afford/is allowed to do this cycle.
        """
        if self.wallet.is_extinct():
            self.alive = False
            return None

        options = self.available_tools()
        if not options:
            logger.info("%s: no affordable/allowed action this cycle", self.name)
            return None

        chosen_name = self._llm_decide(self.genome.strategy_prompt, options, self.genome.temperature)
        tool = self.tools.get(chosen_name)
        if tool is None or tool not in options:
            logger.warning("%s: decision picked unavailable tool %r, skipping cycle", self.name, chosen_name)
            return None

        self.action_policy.check(tool.category)
        if tool.cost > 0:
            self.wallet.debit(tool.cost, f"metabolic cost: {tool.name}")
        result = tool.run()
        logger.info("%s: ran %s for cost %s -> %s", self.name, tool.name, tool.cost, result)

        if self.wallet.is_extinct():
            self.alive = False
        return result

    def _default_llm_decide(self, strategy_prompt: str, options: List[ToolSpec], temperature: float) -> str:
        """Ask Claude which tool to run next via the `claude` CLI in headless
        mode, authenticated with CLAUDE_CODE_OAUTH_TOKEN (a Claude
        subscription token from `claude setup-token`) instead of a metered
        Anthropic API key.

        `temperature` isn't passed through — the CLI doesn't expose sampling
        params — it only shapes Genome mutation in Spawner. Each call spins
        up a full Claude Code session (real latency, session overhead) and
        draws on the subscription's usage limits rather than being billed
        per token, so this fits a slow simulation loop, not a tight one.
        """
        import json
        import subprocess

        if not os.environ.get("CLAUDE_CODE_OAUTH_TOKEN"):
            raise RuntimeError(
                "CLAUDE_CODE_OAUTH_TOKEN is not set. Generate one with `claude setup-token` "
                "(requires a Claude subscription) and export it before running the simulation."
            )

        tool_menu = "\n".join(f"- {tool.name}: {tool.description} (cost={tool.cost})" for tool in options)
        prompt = (
            f"Your current balance is {self.wallet.balance}.\n"
            f"Available tools:\n{tool_menu}\n\n"
            "Reply with only the exact name of the one tool to run next."
        )

        command = ["claude", "-p", prompt, "--append-system-prompt", strategy_prompt, "--output-format", "json"]
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

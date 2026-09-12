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
from dataclasses import dataclass
from decimal import Decimal
from typing import Callable, Dict, List, Optional, Tuple

from core.action_policy import ActionCategory, ActionPolicy
from core.llm_connectors import get_connector
from core.llm_connectors.base import LlmConnector
from core.wallet import Wallet

logger = logging.getLogger(__name__)

# Sentinel an agent's decision can reply with instead of a tool name, to ask
# a human for a capability it doesn't have. This never grants anything by
# itself - see Agent.step() and the module docstring's allowlist note.
TOOL_REQUEST_PREFIX = "REQUEST_TOOL:"


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
        connector: Optional[LlmConnector] = None,
    ) -> None:
        self.name = name
        self.wallet = wallet
        self.action_policy = action_policy
        self.genome = genome
        self.tools = tools
        self.connector = connector or get_connector()
        self._llm_decide = llm_decide or self._default_llm_decide
        self.generation = 0
        self.alive = True
        self.tool_requests: List[str] = []

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

        Returns the tool's output, or None if the agent is extinct, has
        nothing it can afford/is allowed to do, or just asked for a new
        capability instead of using an existing tool.
        """
        if self.wallet.is_extinct():
            self.alive = False
            return None

        options = self.available_tools()
        if not options:
            logger.info("%s: no affordable/allowed action this cycle", self.name)
            return None

        chosen_name = self._llm_decide(self.genome.strategy_prompt, options, self.genome.temperature)

        if chosen_name.startswith(TOOL_REQUEST_PREFIX):
            request = chosen_name[len(TOOL_REQUEST_PREFIX):].strip()
            self.tool_requests.append(request)
            logger.warning("%s is requesting a capability it doesn't have: %s", self.name, request)
            return None

        tool = self.tools.get(chosen_name)
        if tool is None or tool not in options:
            logger.warning("%s: decision picked unavailable tool %r, skipping cycle", self.name, chosen_name)
            return None

        self.action_policy.check(tool.category)
        if tool.cost > 0:
            self.wallet.debit(tool.cost, f"metabolic cost: {tool.name}")

        try:
            result = tool.run()
        except Exception as exc:  # noqa: BLE001 - a tool failing must not take the whole simulation down
            logger.warning("%s: %s raised %s: %s", self.name, tool.name, type(exc).__name__, exc)
            result = None
        else:
            logger.info("%s: ran %s for cost %s -> %s", self.name, tool.name, tool.cost, result)

        if self.wallet.is_extinct():
            self.alive = False
        return result

    def _default_llm_decide(self, strategy_prompt: str, options: List[ToolSpec], temperature: float) -> str:
        """Ask this agent's configured LlmConnector which tool to run next."""
        tool_menu = "\n".join(f"- {tool.name}: {tool.description} (cost={tool.cost})" for tool in options)
        user_prompt = (
            f"Your current balance is {self.wallet.balance}.\n"
            f"Available tools:\n{tool_menu}\n\n"
            "Reply with only the exact name of the one tool to run next.\n"
            "If none of these tools let you pursue your strategy, you may instead reply with "
            f'exactly "{TOOL_REQUEST_PREFIX} <one-line description of the capability you need and why>" '
            "- a human will review this before anything is added; you cannot grant yourself new tools."
        )
        return self.connector.complete(
            system_prompt=strategy_prompt, user_prompt=user_prompt, temperature=temperature
        ).strip()

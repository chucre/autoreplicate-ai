"""Simulation entry point.

Runs a population of agents through decision cycles, spawning children
when their wallets double and retiring them at zero, all against
SimulatedPaymentAdapter — phase 1 never touches a real PSP. See the
project brief for why: the real Pix adapter only comes in once a full
simulated lifecycle has been validated end to end.
"""
from __future__ import annotations

import logging
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Optional

from core.action_policy import ActionCategory, ActionPolicy
from core.agent import Agent, Genome, ToolSpec
from core.payment_adapter import SimulatedPaymentAdapter
from core.spawner import Spawner, SpawnPendingApproval
from core.wallet import Wallet
from tools.hold import hold
from tools.market_data import get_spot_price

LOG_DIR = Path(__file__).parent / "logs"
INITIAL_BALANCE = Decimal("300")
AUTO_APPROVE_LIMIT = Decimal("50")
MAX_CYCLES = 50

logger = logging.getLogger("main")


def build_tool_registry() -> Dict[str, ToolSpec]:
    return {
        "get_spot_price": ToolSpec(
            name="get_spot_price",
            category=ActionCategory.MARKET_DATA_READ,
            cost=Decimal("0.05"),
            run=get_spot_price,
            description="Read the public BTC/BRL spot price.",
        ),
        "hold": ToolSpec(
            name="hold",
            category=ActionCategory.MARKET_DATA_READ,
            cost=Decimal("0"),
            run=hold,
            description="Do nothing this cycle.",
        ),
    }


def build_root_agent(tools: Dict[str, ToolSpec]) -> Agent:
    wallet = Wallet(INITIAL_BALANCE, auto_approve_limit=AUTO_APPROVE_LIMIT)
    policy = ActionPolicy()
    genome = Genome(
        temperature=0.7,
        strategy_prompt=(
            "You are the root agent of a simulated economy. Prefer checking "
            "the market price; only hold when there's nothing useful to do."
        ),
        enabled_tools=tuple(tools.keys()),
    )
    return Agent(name="root", wallet=wallet, action_policy=policy, genome=genome, tools=tools)


def setup_logging() -> None:
    LOG_DIR.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.FileHandler(LOG_DIR / "simulation.log"), logging.StreamHandler()],
    )


def request_spawn_approval(pending: SpawnPendingApproval) -> bool:
    answer = input(
        f"{pending.parent.name} wants to spawn a child with {pending.transfer_amount} "
        "capital. Approve? [y/N] "
    )
    return answer.strip().lower() == "y"


def run_simulation() -> None:
    setup_logging()
    tools = build_tool_registry()
    payment_adapter = SimulatedPaymentAdapter(starting_balance=INITIAL_BALANCE)
    spawner = Spawner(require_human_approval=True)

    population: List[Agent] = [build_root_agent(tools)]
    cycle = 0

    for cycle in range(MAX_CYCLES):
        logger.info("=== cycle %d: %d agent(s) alive ===", cycle, len(population))
        next_population: List[Agent] = []

        for agent in population:
            agent.step()

            if agent.wallet.is_extinct():
                logger.info("%s went extinct at cycle %d", agent.name, cycle)
                continue

            child: Optional[Agent] = None
            try:
                child = spawner.maybe_spawn(agent, tools)
            except SpawnPendingApproval as pending:
                if request_spawn_approval(pending):
                    child = spawner.maybe_spawn(agent, tools, human_approved=True)
                else:
                    logger.info("human declined spawn for %s", pending.parent.name)

            next_population.append(agent)
            if child is not None:
                next_population.append(child)

        population = next_population
        if not population:
            logger.info("population extinct at cycle %d, ending simulation", cycle)
            break

    logger.info("simulation ended after %d cycle(s) with %d agent(s) alive", cycle + 1, len(population))


if __name__ == "__main__":
    run_simulation()

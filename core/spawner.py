"""Spawner: turns a doubled wallet into a new child Agent.

Spawning moves real capital out of the parent's wallet, so — like any
other transfer — it goes through Wallet's own approval gate. On top of
that, `require_human_approval` (on by default) makes the spawn itself stop
and wait for an explicit human OK before a child is created at all,
regardless of the transfer amount. Turn it off only once the full
birth -> operate -> double -> spawn cycle has been validated enough that
later generations don't need a human in the loop for this step.
"""
from __future__ import annotations

import logging
import random
from dataclasses import replace
from decimal import Decimal
from typing import Callable, Dict, Optional

from core.action_policy import ActionPolicy
from core.agent import Agent, Genome, ToolSpec
from core.wallet import Wallet

logger = logging.getLogger(__name__)

MutationFn = Callable[[Genome], Genome]


class SpawnPendingApproval(Exception):
    """Raised when a spawn needs a human's OK before it can proceed.

    Call `maybe_spawn` again with `human_approved=True` once a person has
    signed off on `transfer_amount` and `child_genome`.
    """

    def __init__(self, parent: Agent, transfer_amount: Decimal, child_genome: Genome) -> None:
        self.parent = parent
        self.transfer_amount = transfer_amount
        self.child_genome = child_genome
        super().__init__(f"spawn of {transfer_amount} from {parent.name} needs human approval")


class Spawner:
    def __init__(
        self,
        spawn_fraction: Decimal = Decimal("0.5"),
        require_human_approval: bool = True,
        mutation_fn: Optional[MutationFn] = None,
    ) -> None:
        self.spawn_fraction = spawn_fraction
        self.require_human_approval = require_human_approval
        self._mutate = mutation_fn or self._default_mutate

    def maybe_spawn(
        self,
        parent: Agent,
        tools: Dict[str, ToolSpec],
        *,
        human_approved: bool = False,
    ) -> Optional[Agent]:
        """If the parent's wallet has doubled, create and return a child
        Agent funded from the parent's capital. Returns None if no spawn is
        due yet.
        """
        if not parent.wallet.has_doubled():
            return None

        transfer_amount = (parent.wallet.balance * self.spawn_fraction).quantize(Decimal("0.01"))
        child_genome = self._mutate(parent.genome)

        if self.require_human_approval and not human_approved:
            raise SpawnPendingApproval(parent, transfer_amount, child_genome)

        transferred = parent.wallet.transfer_out(
            transfer_amount, f"spawn child from {parent.name}", approved=human_approved
        )

        child = Agent(
            name=f"{parent.name}.child.{random.randint(1000, 9999)}",
            wallet=Wallet(transferred, auto_approve_limit=parent.wallet.auto_approve_limit),
            action_policy=ActionPolicy(parent.action_policy.allowed_categories),
            genome=child_genome,
            tools=tools,
            model=parent.model,
        )
        child.generation = parent.generation + 1
        logger.info(
            "spawned %s (gen %d) from %s with %s capital", child.name, child.generation, parent.name, transferred
        )
        return child

    @staticmethod
    def _default_mutate(parent_genome: Genome) -> Genome:
        mutated_temperature = min(1.0, max(0.0, parent_genome.temperature + random.uniform(-0.15, 0.15)))
        return replace(parent_genome, temperature=round(mutated_temperature, 3))

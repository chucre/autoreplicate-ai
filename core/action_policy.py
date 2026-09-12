"""ActionPolicy: the allowlist of action categories an agent may ever attempt.

This is deliberately an ALLOWLIST, not a blocklist. A blocklist only stops
actions someone thought to name in advance; every new tool added later
would run by default unless someone remembered to blocklist it. An
allowlist inverts that: nothing an agent does reaches a real tool unless
its category is already here, which means a human looked at what that
category actually permits before any agent could use it.

Adding a category is a manual, reviewed act on its own — never something
an agent, a spawner, or a "the LLM suggested it" code path can do. If you
add one, write down here what it actually permits and why it's safe to
hand to an autonomous agent.
"""
from __future__ import annotations

from enum import Enum
from typing import FrozenSet, Optional


class ActionCategory(str, Enum):
    # Read-only. No funds move, no external state changes. Safe by
    # construction — this exists mainly so market-data tools still have to
    # declare a category and go through the same gate as everything else.
    MARKET_DATA_READ = "MARKET_DATA_READ"

    # Places real spot trades using the agent's own wallet funds on an
    # exchange account scoped to that wallet. Never margin/leverage, never
    # a market order sized beyond the wallet's own balance.
    EXCHANGE_TRADE_SPOT = "EXCHANGE_TRADE_SPOT"

    # Lets the agent pick up and complete paid gig-work tasks from a task
    # marketplace API. No ability to *post* tasks or move money outside the
    # marketplace's own escrow/payout flow.
    TASK_MARKETPLACE = "TASK_MARKETPLACE"


ALLOWLIST: FrozenSet[ActionCategory] = frozenset(
    {
        ActionCategory.MARKET_DATA_READ,
        ActionCategory.EXCHANGE_TRADE_SPOT,
        ActionCategory.TASK_MARKETPLACE,
    }
)


class ActionNotAllowedError(Exception):
    """Raised when an action's category isn't on the allowlist."""


class ActionPolicy:
    """Gate every tool call must pass through before it can spend money or
    touch anything external."""

    def __init__(self, allowed_categories: Optional[FrozenSet[ActionCategory]] = None) -> None:
        self.allowed_categories = ALLOWLIST if allowed_categories is None else allowed_categories

    def is_allowed(self, category: ActionCategory) -> bool:
        return category in self.allowed_categories

    def check(self, category: ActionCategory) -> None:
        if not self.is_allowed(category):
            raise ActionNotAllowedError(f"{category} is not on the allowlist; refusing to run this action")

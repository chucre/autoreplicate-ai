"""Wallet: tracks an agent's balance and enforces spending/approval limits.

Amounts are Decimal, quantized to cents, since this eventually backs a
real Pix transfer and float drift is not acceptable for money.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import ROUND_HALF_UP, Decimal
from typing import Optional, Union

Amount = Union[Decimal, float, str, int]

_TWO_PLACES = Decimal("0.01")


def _quantize(amount: Decimal) -> Decimal:
    return amount.quantize(_TWO_PLACES, rounding=ROUND_HALF_UP)


class InsufficientFundsError(Exception):
    """Raised when a debit or transfer would take the wallet negative."""


class ApprovalRequiredError(Exception):
    """Raised when a transaction is above auto_approve_limit and wasn't
    passed approved=True by a human-facing caller."""


@dataclass
class Transaction:
    kind: str  # "debit" | "credit" | "transfer_out"
    amount: Decimal
    description: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    approved_by_human: Optional[bool] = None


class Wallet:
    """An agent's real (or simulated) balance.

    `auto_approve_limit` is the ceiling strictly above which a transaction
    needs a human to pass `approved=True` explicitly. The wallet itself
    never grants that approval — it only enforces that someone did.
    """

    def __init__(self, initial_balance: Amount, auto_approve_limit: Amount = Decimal("50")) -> None:
        self.initial_balance = _quantize(Decimal(str(initial_balance)))
        self.balance = self.initial_balance
        self.auto_approve_limit = _quantize(Decimal(str(auto_approve_limit)))
        self.history: list[Transaction] = []

    def debit(self, amount: Amount, description: str, *, approved: bool = False) -> None:
        amount = _quantize(Decimal(str(amount)))
        self._require_positive(amount)
        self._check_approval(amount, approved)
        if amount > self.balance:
            raise InsufficientFundsError(f"cannot debit {amount}: balance is only {self.balance}")
        self.balance -= amount
        self.history.append(Transaction("debit", amount, description, approved_by_human=approved or None))

    def credit(self, amount: Amount, description: str) -> None:
        amount = _quantize(Decimal(str(amount)))
        self._require_positive(amount)
        self.balance += amount
        self.history.append(Transaction("credit", amount, description))

    def transfer_out(self, amount: Amount, description: str, *, approved: bool = False) -> Decimal:
        """Move capital out of this wallet (e.g. to fund a spawned child).

        Subject to the same approval gate as `debit`. Returns the
        transferred amount so the caller can credit it elsewhere.
        """
        amount = _quantize(Decimal(str(amount)))
        self._require_positive(amount)
        self._check_approval(amount, approved)
        if amount > self.balance:
            raise InsufficientFundsError(f"cannot transfer {amount}: balance is only {self.balance}")
        self.balance -= amount
        self.history.append(Transaction("transfer_out", amount, description, approved_by_human=approved or None))
        return amount

    def has_doubled(self) -> bool:
        return self.balance >= self.initial_balance * 2

    def is_extinct(self) -> bool:
        return self.balance <= 0

    def _check_approval(self, amount: Decimal, approved: bool) -> None:
        if amount > self.auto_approve_limit and not approved:
            raise ApprovalRequiredError(
                f"{amount} is above auto_approve_limit ({self.auto_approve_limit}); "
                "requires explicit human approval (approved=True)"
            )

    @staticmethod
    def _require_positive(amount: Decimal) -> None:
        if amount <= 0:
            raise ValueError("amount must be positive")

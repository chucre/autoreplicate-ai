"""PaymentAdapter: abstraction over the real-money settlement layer.

Phase 1 only ever wires up SimulatedPaymentAdapter. A real Pix-backed
adapter (Banco Inter, Mercado Pago, EFI/Gerencianet — plain bank accounts
generally don't expose automated Pix) is a separate, later implementation
of this same interface, swapped in only once a full simulated lifecycle
(birth -> operate -> double -> spawn, or extinction) has been validated.

This layer only ever moves money to a destination it's told to. It does
not decide whether an amount needs human approval, whether a category is
allowed, or whether a wallet has doubled — that all lives above it, in
Wallet / ActionPolicy / Spawner. In particular, nothing here should ever
be called with a destination Pix key that wasn't already known-good; see
the project's non-negotiable rules about arbitrary/unknown keys.
"""
from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Union

Amount = Union[Decimal, float, str, int]


@dataclass
class PaymentResult:
    success: bool
    reference: str
    amount: Decimal
    detail: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class PaymentAdapter(ABC):
    """Interface a real PSP integration must implement."""

    @abstractmethod
    def send(self, *, destination_key: str, amount: Amount, description: str) -> PaymentResult:
        """Send `amount` to a known Pix key."""

    @abstractmethod
    def get_balance(self) -> Decimal:
        """Return the real, settled balance backing this adapter."""


class SimulatedPaymentAdapter(PaymentAdapter):
    """In-memory PSP stand-in: no network call, no real money ever moves.

    Keeps its own ledger so the whole agent lifecycle can be exercised
    safely before any real adapter exists.
    """

    def __init__(self, starting_balance: Amount = Decimal("0")) -> None:
        self._balance = Decimal(str(starting_balance))
        self.ledger: list[PaymentResult] = []

    def send(self, *, destination_key: str, amount: Amount, description: str) -> PaymentResult:
        amount = Decimal(str(amount))
        if amount <= 0:
            raise ValueError("amount must be positive")

        if amount > self._balance:
            result = PaymentResult(
                success=False,
                reference=str(uuid.uuid4()),
                amount=amount,
                detail=f"insufficient simulated balance ({self._balance}) to send to {destination_key}: {description}",
            )
        else:
            self._balance -= amount
            result = PaymentResult(
                success=True,
                reference=str(uuid.uuid4()),
                amount=amount,
                detail=f"simulated send to {destination_key}: {description}",
            )

        self.ledger.append(result)
        return result

    def get_balance(self) -> Decimal:
        return self._balance

    def credit(self, amount: Amount) -> None:
        """Test/setup helper: fund the simulated account directly (e.g. to
        seed a newly spawned child). Never available on a real adapter."""
        amount = Decimal(str(amount))
        if amount <= 0:
            raise ValueError("amount must be positive")
        self._balance += amount

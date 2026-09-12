from decimal import Decimal

import pytest

from core.wallet import ApprovalRequiredError, InsufficientFundsError, Wallet


def test_debit_reduces_balance():
    wallet = Wallet(100)
    wallet.debit(10, "test")
    assert wallet.balance == Decimal("90.00")


def test_debit_rejects_insufficient_funds():
    wallet = Wallet(10)
    with pytest.raises(InsufficientFundsError):
        wallet.debit(20, "test")


def test_debit_rejects_non_positive_amount():
    wallet = Wallet(10)
    with pytest.raises(ValueError):
        wallet.debit(0, "test")


def test_credit_increases_balance():
    wallet = Wallet(100)
    wallet.credit(25, "test")
    assert wallet.balance == Decimal("125.00")


def test_debit_at_auto_approve_limit_does_not_need_approval():
    wallet = Wallet(200, auto_approve_limit=50)
    wallet.debit(50, "exactly at the limit")
    assert wallet.balance == Decimal("150.00")


def test_debit_above_auto_approve_limit_requires_approval():
    wallet = Wallet(200, auto_approve_limit=50)
    with pytest.raises(ApprovalRequiredError):
        wallet.debit(60, "big spend")


def test_debit_above_limit_succeeds_when_approved():
    wallet = Wallet(200, auto_approve_limit=50)
    wallet.debit(60, "big spend", approved=True)
    assert wallet.balance == Decimal("140.00")


def test_transfer_out_below_limit_does_not_need_approval():
    wallet = Wallet(200, auto_approve_limit=50)
    transferred = wallet.transfer_out(30, "spawn child")
    assert transferred == Decimal("30.00")
    assert wallet.balance == Decimal("170.00")


def test_transfer_out_above_limit_requires_approval():
    wallet = Wallet(200, auto_approve_limit=50)
    with pytest.raises(ApprovalRequiredError):
        wallet.transfer_out(51, "spawn child")


def test_has_doubled():
    wallet = Wallet(100)
    assert not wallet.has_doubled()
    wallet.credit(100, "profit")
    assert wallet.has_doubled()


def test_is_extinct():
    wallet = Wallet(10)
    assert not wallet.is_extinct()
    wallet.debit(10, "spend it all")
    assert wallet.is_extinct()


def test_history_records_transactions():
    wallet = Wallet(100)
    wallet.debit(10, "cost")
    wallet.credit(5, "refund")
    assert [entry.kind for entry in wallet.history] == ["debit", "credit"]

import pytest

from core.action_policy import ALLOWLIST, ActionCategory, ActionNotAllowedError, ActionPolicy


def test_default_policy_allows_every_listed_category():
    policy = ActionPolicy()
    for category in ALLOWLIST:
        assert policy.is_allowed(category)
        policy.check(category)  # must not raise


def test_default_policy_matches_every_defined_category():
    # Guards against silently adding a category to the enum without also
    # reviewing it onto (or deliberately keeping it off) the allowlist.
    assert set(ActionCategory) == set(ALLOWLIST)


def test_policy_rejects_category_outside_its_own_allowlist():
    policy = ActionPolicy(allowed_categories=frozenset({ActionCategory.MARKET_DATA_READ}))
    assert not policy.is_allowed(ActionCategory.EXCHANGE_TRADE_SPOT)
    with pytest.raises(ActionNotAllowedError):
        policy.check(ActionCategory.EXCHANGE_TRADE_SPOT)


def test_empty_policy_rejects_everything():
    policy = ActionPolicy(allowed_categories=frozenset())
    for category in ActionCategory:
        assert not policy.is_allowed(category)
        with pytest.raises(ActionNotAllowedError):
            policy.check(category)

"""MARKET_DATA_READ tools: public, read-only price data.

These never move money and never touch the wallet directly — the
metabolic cost of calling one is charged by Agent.step(), not by the tool
itself. This module makes a real network call to a public endpoint; no
API key or account is involved.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request

_COINGECKO_SIMPLE_PRICE_URL = "https://api.coingecko.com/api/v3/simple/price"


def get_spot_price(coin_id: str = "bitcoin", vs_currency: str = "brl", timeout: float = 5.0) -> str:
    """Fetch a public spot price. CoinGecko's free tier rate-limits fairly
    aggressively, so a failure here (429, timeout, DNS hiccup, ...) is an
    expected outcome of calling a public API, not a bug — return a message
    describing it instead of raising, so one flaky call doesn't take down
    the agent that made it.
    """
    url = f"{_COINGECKO_SIMPLE_PRICE_URL}?ids={coin_id}&vs_currencies={vs_currency}"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        return f"failed to fetch {coin_id}/{vs_currency}: {exc}"

    price = data.get(coin_id, {}).get(vs_currency)
    if price is None:
        return f"no price returned for {coin_id}/{vs_currency}"
    return f"{coin_id}/{vs_currency} = {price}"

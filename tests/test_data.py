from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pandas as pd

from trade_bot.data import fetch_crypto_ohlcv, fetch_ohlcv


def _fake_client(captured_requests: list, df: pd.DataFrame | None = None) -> MagicMock:
    client = MagicMock()

    def get_stock_bars(request):
        captured_requests.append(request)
        result = MagicMock()
        result.df = df if df is not None else pd.DataFrame()
        return result

    client.get_stock_bars.side_effect = get_stock_bars
    return client


def test_without_since_start_reaches_back_far_enough_for_max_bars():
    # Alpaca's API only returns bars from "today" when `start` is omitted -
    # `limit` alone does NOT reach further back (discovered because Jef's
    # live paper trading never warmed up its indicators: it always got only
    # today's handful of H1 bars, never the 150+ needed for slow_ma/rsi/
    # trend_ma, so entry_signal could never fire).
    captured: list = []
    client = _fake_client(captured)

    fetch_ohlcv(client, "AAPL", "H1", since=None, max_bars=600)

    request = captured[0]
    assert request.start is not None
    # H1 has ~7 bars/trading day, so 600 bars needs multiple months of
    # calendar time, not "today". (Alpaca's request model stores `start` as
    # a naive datetime internally, so compare naive-to-naive here.)
    assert request.start < datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=30)


def test_with_explicit_since_is_used_as_is():
    captured: list = []
    client = _fake_client(captured)
    since = datetime(2023, 1, 1, tzinfo=timezone.utc)

    fetch_ohlcv(client, "AAPL", "H1", since=since, max_bars=600)

    assert captured[0].start == since.replace(tzinfo=None)


def test_without_since_does_not_cap_the_request_by_max_bars():
    # Alpaca returns bars chronologically FORWARD from `start` up to
    # `limit` bars - capping `limit` here would return the OLDEST bars in
    # the estimated range, not the most recent ones (discovered live: a
    # request for 600 H1 bars landed on bars from three months ago instead
    # of today, because `limit=600` from a computed start three months back
    # returned the oldest 600 bars in that window, not the newest).
    captured: list = []
    client = _fake_client(captured)

    fetch_ohlcv(client, "AAPL", "H1", since=None, max_bars=600)

    assert captured[0].limit is None


def test_with_explicit_since_still_caps_by_max_bars():
    captured: list = []
    client = _fake_client(captured)

    fetch_ohlcv(client, "AAPL", "H1", since=datetime(2023, 1, 1, tzinfo=timezone.utc), max_bars=600)

    assert captured[0].limit == 600


def test_without_since_trims_client_side_to_the_most_recent_bars():
    idx = pd.date_range("2020-01-01", periods=1000, freq="h", tz="UTC")
    df = pd.DataFrame(
        {"open": range(1000), "high": range(1000), "low": range(1000), "close": range(1000), "volume": [1.0] * 1000},
        index=idx,
    )
    client = _fake_client([], df=df)

    result = fetch_ohlcv(client, "AAPL", "H1", since=None, max_bars=100)

    assert len(result) == 100
    assert result["close"].iloc[-1] == 999  # the most recent bar, not the oldest


def test_without_since_and_without_max_bars_still_reaches_back():
    # The paper trader's default digest/live path and any other caller that
    # omits both `since` and `max_bars` must not silently fall back to
    # "today only" either.
    captured: list = []
    client = _fake_client(captured)

    fetch_ohlcv(client, "AAPL", "H1", since=None, max_bars=None)

    assert captured[0].start is not None
    assert captured[0].start < datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=5)


def _fake_crypto_client(captured_requests: list) -> MagicMock:
    client = MagicMock()

    def get_crypto_bars(request):
        captured_requests.append(request)
        result = MagicMock()
        result.df = pd.DataFrame()
        return result

    client.get_crypto_bars.side_effect = get_crypto_bars
    return client


def test_crypto_without_since_reaches_back_far_enough_for_max_bars():
    # Crypto trades 24/7 - no weekend/holiday gaps like stocks, but still
    # needs an explicit `start` for the same reason fetch_ohlcv does.
    captured: list = []
    client = _fake_crypto_client(captured)

    fetch_crypto_ohlcv(client, "BTC/USD", "H1", since=None, max_bars=600)

    request = captured[0]
    assert request.start is not None
    # H1 has 24 bars/day for crypto, so 600 bars needs ~25 days, not "today".
    assert request.start < datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=15)


def test_crypto_without_since_does_not_cap_the_request_by_max_bars():
    captured: list = []
    client = _fake_crypto_client(captured)

    fetch_crypto_ohlcv(client, "BTC/USD", "H1", since=None, max_bars=600)

    assert captured[0].limit is None


def test_crypto_with_explicit_since_is_used_as_is():
    captured: list = []
    client = _fake_crypto_client(captured)
    since = datetime(2023, 1, 1, tzinfo=timezone.utc)

    fetch_crypto_ohlcv(client, "BTC/USD", "H1", since=since, max_bars=600)

    assert captured[0].start == since.replace(tzinfo=None)

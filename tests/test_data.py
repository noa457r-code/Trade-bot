from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pandas as pd

from trade_bot.data import fetch_ohlcv


def _fake_client(captured_requests: list) -> MagicMock:
    client = MagicMock()

    def get_stock_bars(request):
        captured_requests.append(request)
        result = MagicMock()
        result.df = pd.DataFrame()
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


def test_without_since_and_without_max_bars_still_reaches_back():
    # The paper trader's default digest/live path and any other caller that
    # omits both `since` and `max_bars` must not silently fall back to
    # "today only" either.
    captured: list = []
    client = _fake_client(captured)

    fetch_ohlcv(client, "AAPL", "H1", since=None, max_bars=None)

    assert captured[0].start is not None
    assert captured[0].start < datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=5)

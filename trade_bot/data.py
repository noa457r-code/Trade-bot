from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import pandas as pd
from alpaca.data.historical.crypto import CryptoHistoricalDataClient
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest, StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

GRANULARITY_MAP: dict[str, TimeFrame] = {
    "M1": TimeFrame(1, TimeFrameUnit.Minute),
    "M5": TimeFrame(5, TimeFrameUnit.Minute),
    "M15": TimeFrame(15, TimeFrameUnit.Minute),
    "M30": TimeFrame(30, TimeFrameUnit.Minute),
    "H1": TimeFrame(1, TimeFrameUnit.Hour),
    "H4": TimeFrame(4, TimeFrameUnit.Hour),
    "D": TimeFrame(1, TimeFrameUnit.Day),
    "W": TimeFrame(1, TimeFrameUnit.Week),
}

# Roughly how many bars a full NYSE trading day produces per granularity
# (6.5h session). Used only to estimate how far back `start` needs to reach
# to cover a bar count - doesn't need to be exact.
BARS_PER_TRADING_DAY: dict[str, float] = {
    "M1": 390, "M5": 78, "M15": 26, "M30": 13, "H1": 7, "H4": 2, "D": 1, "W": 1 / 5,
}


def _default_start_for_bar_count(granularity: str, max_bars: int) -> datetime:
    """Alpaca's API only returns bars from "today" when `start` is omitted -
    `limit` alone does NOT reach further back. Without an explicit `start`,
    this computes one far enough in the past to actually cover `max_bars`
    bars, with a safety margin for weekends/holidays when markets are
    closed. (Discovered because paper trading always calls fetch_ohlcv
    without `since` - with no explicit start, it only ever got today's
    handful of bars, never enough to warm up slow_ma/rsi/trend_ma, so
    entry_signal could never fire.)
    """
    calendar_days = math.ceil(max_bars / BARS_PER_TRADING_DAY[granularity] * 1.6) + 5
    return datetime.now(timezone.utc) - timedelta(days=calendar_days)


# Crypto trades 24/7 - no weekend/holiday gaps, so far more bars per
# calendar day than the equivalent stock granularity.
CRYPTO_BARS_PER_DAY: dict[str, float] = {
    "M1": 1440, "M5": 288, "M15": 96, "M30": 48, "H1": 24, "H4": 6, "D": 1, "W": 1 / 7,
}


def _default_start_for_crypto_bar_count(granularity: str, max_bars: int) -> datetime:
    calendar_days = math.ceil(max_bars / CRYPTO_BARS_PER_DAY[granularity] * 1.2) + 2
    return datetime.now(timezone.utc) - timedelta(days=calendar_days)


def make_client(api_key: str, secret_key: str) -> StockHistoricalDataClient:
    return StockHistoricalDataClient(api_key, secret_key)


def make_crypto_client(api_key: str, secret_key: str) -> CryptoHistoricalDataClient:
    return CryptoHistoricalDataClient(api_key, secret_key)


def fetch_ohlcv(
    client: StockHistoricalDataClient,
    instrument: str,
    granularity: str,
    since: datetime | None = None,
    max_bars: int | None = None,
) -> pd.DataFrame:
    """Fetch Alpaca OHLCV bars for a single stock symbol.

    Without `since`, returns the most recent `max_bars` (or 500) bars up to
    now. With `since`, returns bars from that date onward (capped at
    `max_bars` if given). Alpaca paginates internally, so no manual paging
    is needed here.

    Returns a DataFrame indexed by UTC timestamp with columns
    open, high, low, close, volume.
    """
    start = since
    if start is None:
        start = _default_start_for_bar_count(granularity, max_bars or 500)

    request = StockBarsRequest(
        symbol_or_symbols=instrument,
        timeframe=GRANULARITY_MAP[granularity],
        start=start,
        limit=max_bars or (None if since else 500),
    )
    bars = client.get_stock_bars(request)
    df = bars.df

    if df.empty:
        empty = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        empty.index.name = "timestamp"
        return empty

    if isinstance(df.index, pd.MultiIndex):
        df = df.xs(instrument, level=0)

    df = df[["open", "high", "low", "close", "volume"]].sort_index()
    df.index = pd.to_datetime(df.index, utc=True)
    df.index.name = "timestamp"

    if max_bars is not None:
        df = df.iloc[-max_bars:]

    return df


def fetch_crypto_ohlcv(
    client: CryptoHistoricalDataClient,
    instrument: str,
    granularity: str,
    since: datetime | None = None,
    max_bars: int | None = None,
) -> pd.DataFrame:
    """Fetch Alpaca OHLCV bars for a crypto pair (e.g. "BTC/USD") - same
    contract as `fetch_ohlcv`, but crypto uses a separate client/request
    type and trades 24/7 (no weekend/holiday gaps to account for).
    """
    start = since
    if start is None:
        start = _default_start_for_crypto_bar_count(granularity, max_bars or 500)

    request = CryptoBarsRequest(
        symbol_or_symbols=instrument,
        timeframe=GRANULARITY_MAP[granularity],
        start=start,
        limit=max_bars or (None if since else 500),
    )
    bars = client.get_crypto_bars(request)
    df = bars.df

    if df.empty:
        empty = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        empty.index.name = "timestamp"
        return empty

    if isinstance(df.index, pd.MultiIndex):
        df = df.xs(instrument, level=0)

    df = df[["open", "high", "low", "close", "volume"]].sort_index()
    df.index = pd.to_datetime(df.index, utc=True)
    df.index.name = "timestamp"

    if max_bars is not None:
        df = df.iloc[-max_bars:]

    return df

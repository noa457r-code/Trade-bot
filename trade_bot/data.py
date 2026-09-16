from __future__ import annotations

import time

import ccxt
import pandas as pd


def make_exchange(exchange_id: str, api_key: str | None = None, api_secret: str | None = None) -> ccxt.Exchange:
    exchange_cls = getattr(ccxt, exchange_id)
    exchange = exchange_cls(
        {
            "apiKey": api_key or "",
            "secret": api_secret or "",
            "enableRateLimit": True,
        }
    )
    return exchange


def fetch_ohlcv(
    exchange: ccxt.Exchange,
    symbol: str,
    timeframe: str,
    since_ms: int | None = None,
    limit: int = 1000,
    max_bars: int | None = None,
) -> pd.DataFrame:
    """Fetch OHLCV candles, paging forward from `since_ms` until caught up.

    Returns a DataFrame indexed by UTC timestamp with columns
    open, high, low, close, volume.
    """
    all_rows: list[list[float]] = []
    cursor = since_ms

    while True:
        batch = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=cursor, limit=limit)
        if not batch:
            break
        all_rows.extend(batch)

        if len(batch) < limit:
            break
        if max_bars is not None and len(all_rows) >= max_bars:
            break

        last_ts = batch[-1][0]
        cursor = last_ts + 1
        time.sleep(exchange.rateLimit / 1000)

    if max_bars is not None:
        all_rows = all_rows[-max_bars:]

    df = pd.DataFrame(all_rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.drop_duplicates(subset="timestamp").set_index("timestamp").sort_index()
    return df

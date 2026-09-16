from __future__ import annotations

from datetime import datetime

import pandas as pd
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
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


def make_client(api_key: str, secret_key: str) -> StockHistoricalDataClient:
    return StockHistoricalDataClient(api_key, secret_key)


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
    request = StockBarsRequest(
        symbol_or_symbols=instrument,
        timeframe=GRANULARITY_MAP[granularity],
        start=since,
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

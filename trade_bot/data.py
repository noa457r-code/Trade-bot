from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone

import pandas as pd
from oandapyV20 import API
from oandapyV20.endpoints.instruments import InstrumentsCandles

GRANULARITY_SECONDS = {
    "S5": 5, "S10": 10, "S15": 15, "S30": 30,
    "M1": 60, "M2": 120, "M4": 240, "M5": 300, "M10": 600, "M15": 900, "M30": 1800,
    "H1": 3600, "H2": 7200, "H3": 10800, "H4": 14400, "H6": 21600, "H8": 28800, "H12": 43200,
    "D": 86400, "W": 604800,
}

OANDA_MAX_COUNT = 5000


def make_client(environment: str, api_token: str) -> API:
    return API(access_token=api_token, environment=environment)


def _parse_candle_time(raw: str) -> datetime:
    return datetime.strptime(raw[:19], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)


def fetch_ohlcv(
    client: API,
    instrument: str,
    granularity: str,
    since: datetime | None = None,
    max_bars: int | None = None,
) -> pd.DataFrame:
    """Fetch OANDA mid-price candles.

    Without `since`, returns the most recent `max_bars` (or 500) completed
    candles up to now. With `since`, pages forward from that point until
    caught up (or `max_bars` is reached).

    Returns a DataFrame indexed by UTC timestamp with columns
    open, high, low, close, volume. Still-forming (incomplete) candles are
    dropped.
    """
    if since is None:
        count = min(max_bars or 500, OANDA_MAX_COUNT)
        params = {"granularity": granularity, "price": "M", "count": count}
        request = InstrumentsCandles(instrument=instrument, params=params)
        client.request(request)
        rows = [c for c in request.response["candles"] if c["complete"]]
    else:
        rows = []
        cursor = since
        step = timedelta(seconds=GRANULARITY_SECONDS[granularity])

        while True:
            params = {
                "granularity": granularity,
                "price": "M",
                "count": OANDA_MAX_COUNT,
                "from": cursor.strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
            request = InstrumentsCandles(instrument=instrument, params=params)
            client.request(request)
            candles = request.response["candles"]
            if not candles:
                break

            rows.extend(c for c in candles if c["complete"])

            if len(candles) < OANDA_MAX_COUNT:
                break
            if max_bars is not None and len(rows) >= max_bars:
                break

            next_cursor = _parse_candle_time(candles[-1]["time"]) + step
            if next_cursor <= cursor:
                break
            cursor = next_cursor
            time.sleep(0.2)

    if max_bars is not None:
        rows = rows[-max_bars:]

    records = [
        {
            "timestamp": c["time"],
            "open": float(c["mid"]["o"]),
            "high": float(c["mid"]["h"]),
            "low": float(c["mid"]["l"]),
            "close": float(c["mid"]["c"]),
            "volume": float(c["volume"]),
        }
        for c in rows
    ]

    df = pd.DataFrame(records, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.drop_duplicates(subset="timestamp").set_index("timestamp").sort_index()
    return df

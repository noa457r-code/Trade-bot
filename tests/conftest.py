import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_ohlcv() -> pd.DataFrame:
    """Deterministic synthetic price series: an uptrend, then a downtrend,
    enough bars to exercise both MA crossover directions.
    """
    rng = np.random.default_rng(seed=42)
    n = 300
    idx = pd.date_range("2023-01-01", periods=n, freq="h", tz="UTC")

    trend = np.concatenate([np.linspace(0, 50, n // 2), np.linspace(50, 0, n - n // 2)])
    # Oscillation on top of the trend so the fast/slow MA cross multiple
    # times, with pullbacks that let RSI cool off before each up-cross.
    oscillation = 8 * np.sin(np.linspace(0, 10 * np.pi, n))
    noise = rng.normal(0, 0.5, n)
    close = 100 + trend + oscillation + noise

    df = pd.DataFrame(
        {
            "open": close - 0.1,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": rng.uniform(10, 100, n),
        },
        index=idx,
    )
    return df

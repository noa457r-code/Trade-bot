from __future__ import annotations

import itertools
from dataclasses import dataclass

import pandas as pd

from trade_bot.strategy import atr, sma


@dataclass
class BollingerConfig:
    period: int = 20
    num_std: float = 2.0
    atr_period: int = 14


def generate_signals(df: pd.DataFrame, cfg: BollingerConfig) -> pd.DataFrame:
    """Bollinger-Band mean-reversion strategy, long-only - the opposite bet
    from the trend-following strategies: buys an oversold dip and exits once
    price reverts back to the mean, instead of riding a trend.

    Long entry: close crosses below the lower band (fresh oversold reading,
                not just "still below" on every bar after).
    Exit:       close crosses back above the middle band (the mean) -
                the reversion this strategy is betting on has played out.
    """
    out = df.copy()
    out["atr"] = atr(out, cfg.atr_period)

    middle = sma(out["close"], cfg.period)
    std = out["close"].rolling(window=cfg.period, min_periods=cfg.period).std()
    out["bb_middle"] = middle
    out["bb_lower"] = middle - cfg.num_std * std
    out["bb_upper"] = middle + cfg.num_std * std

    below_lower = out["close"] < out["bb_lower"]
    crossed_below_lower = below_lower & ~below_lower.shift(1, fill_value=False)

    above_middle = out["close"] > middle
    crossed_above_middle = above_middle & ~above_middle.shift(1, fill_value=False)

    valid = out["atr"].notna() & middle.notna() & std.notna()
    out["entry_signal"] = crossed_below_lower & valid
    out["exit_signal"] = crossed_above_middle & valid

    return out


# Parameter grid for walk-forward validation, plus the same ATR stop/target
# multipliers used for the other strategies' grids so results are comparable.
PARAM_GRID: list[dict] = [
    {"period": period, "num_std": num_std, "stop_loss_atr_mult": sl_mult, "take_profit_atr_mult": tp_mult}
    for period, num_std, sl_mult, tp_mult in itertools.product(
        [10, 20, 30],
        [1.5, 2.0, 2.5],
        [1.5, 2.0, 3.0],
        [3.0, 4.0, 6.0],
    )
]

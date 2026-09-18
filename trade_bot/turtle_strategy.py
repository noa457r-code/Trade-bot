from __future__ import annotations

import itertools
from dataclasses import dataclass

import pandas as pd

from trade_bot.strategy import atr


@dataclass
class TurtleConfig:
    entry_channel: int = 20  # N-day high breakout triggers entry (classic Turtle System 1)
    exit_channel: int = 10   # N-day low breakout triggers exit
    atr_period: int = 14


# Parameter grid for walk-forward validation - entry/exit channel pairs plus
# the same ATR stop/target multipliers used for the sma_rsi grid, so results
# are comparable.
PARAM_GRID: list[dict] = [
    {"entry_channel": entry, "exit_channel": exit_, "stop_loss_atr_mult": sl_mult, "take_profit_atr_mult": tp_mult}
    for (entry, exit_), sl_mult, tp_mult in itertools.product(
        [(10, 5), (20, 10), (55, 20)],
        [1.5, 2.0, 3.0],
        [3.0, 4.0, 6.0],
    )
]


def generate_signals(df: pd.DataFrame, cfg: TurtleConfig) -> pd.DataFrame:
    """Donchian-channel breakout strategy ("Turtle Trading" System 1),
    long-only - no shorts or pyramiding, consistent with the rest of the bot.

    Long entry: close breaks above the highest high of the prior
                `entry_channel` bars (today's own bar excluded, so there's
                no lookahead).
    Exit:       close breaks below the lowest low of the prior
                `exit_channel` bars.
    """
    out = df.copy()
    out["atr"] = atr(out, cfg.atr_period)

    prior_entry_high = out["high"].rolling(window=cfg.entry_channel, min_periods=cfg.entry_channel).max().shift(1)
    prior_exit_low = out["low"].rolling(window=cfg.exit_channel, min_periods=cfg.exit_channel).min().shift(1)

    out["entry_signal"] = out["close"] > prior_entry_high
    out["exit_signal"] = out["close"] < prior_exit_low

    valid = out["atr"].notna() & prior_entry_high.notna() & prior_exit_low.notna()
    out["entry_signal"] &= valid
    out["exit_signal"] &= valid

    return out

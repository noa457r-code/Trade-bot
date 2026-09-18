from __future__ import annotations

import pandas as pd

from trade_bot.config import RiskConfig
from trade_bot.momentum_strategy import (
    MomentumConfig,
    _rank_top_momentum,
    run_momentum_backtest,
)


def _risk_cfg(**overrides) -> RiskConfig:
    defaults = dict(
        initial_capital=10000.0, risk_per_trade=0.01, stop_loss_atr_mult=1.5,
        take_profit_atr_mult=4.0, fee_pct=0.0,
    )
    defaults.update(overrides)
    return RiskConfig(**defaults)


def _make_df(closes: list[float], lows: list[float] | None = None) -> pd.DataFrame:
    idx = pd.date_range("2023-01-01", periods=len(closes), freq="h", tz="UTC")
    lows = lows if lows is not None else closes
    return pd.DataFrame(
        {"open": closes, "high": closes, "low": lows, "close": closes, "volume": [10.0] * len(closes)},
        index=idx,
    )


def test_rank_top_momentum_picks_highest_positive_returns():
    now = {"A": 120.0, "B": 110.0, "C": 90.0}
    then = {"A": 100.0, "B": 100.0, "C": 100.0}  # A +20%, B +10%, C -10%

    ranked = _rank_top_momentum(now, then, top_k=2)

    assert ranked == ["A", "B"]


def test_rank_top_momentum_excludes_negative_momentum():
    now = {"A": 120.0, "B": 90.0, "C": 80.0}
    then = {"A": 100.0, "B": 100.0, "C": 100.0}  # only A is positive

    ranked = _rank_top_momentum(now, then, top_k=3)

    assert ranked == ["A"]


def test_opens_positions_only_in_top_k_after_lookback():
    # A trends up fast, B trends up slowly, C trends down - over a 5-bar
    # lookback, A should clearly rank above both B and C.
    price_data = {
        "A": _make_df([100, 105, 110, 115, 120, 125]),
        "B": _make_df([100, 101, 102, 103, 104, 105]),
        "C": _make_df([100, 98, 96, 94, 92, 90]),
    }
    cfg = MomentumConfig(lookback_bars=5, rebalance_bars=1000, top_k=1, atr_period=2, stop_loss_atr_mult=5.0)

    result = run_momentum_backtest(price_data, _risk_cfg(), cfg)

    opened_instruments = {t.instrument for t in result.trades}
    assert opened_instruments == {"A"}


def test_stop_loss_closes_position_between_rebalances():
    # A looks like the momentum winner at the rebalance bar, then drops
    # sharply on the very next bar - the stop-loss should close it before
    # the next scheduled rebalance (rebalance_bars is set far out).
    price_data = {
        "A": _make_df([100, 105, 110, 115, 120, 60], lows=[100, 105, 110, 115, 120, 55]),
        "B": _make_df([100, 100, 100, 100, 100, 100]),
    }
    cfg = MomentumConfig(lookback_bars=4, rebalance_bars=1000, top_k=1, atr_period=2, stop_loss_atr_mult=1.0)

    result = run_momentum_backtest(price_data, _risk_cfg(), cfg)

    stopped = [t for t in result.trades if t.exit_reason == "stop_loss"]
    assert len(stopped) == 1
    assert stopped[0].instrument == "A"


def test_position_closed_on_rebalance_when_dropped_from_top_k():
    # A wins the first rebalance (i=4). By the second rebalance (i=8), A is
    # still rising (never hits its stop) but C has overtaken it by a wide
    # margin - A should be closed with exit_reason "rebalance", not a stop.
    a = [100, 110, 120, 130, 140, 142, 144, 146, 148]
    b = [100, 100, 100, 100, 100, 100, 100, 100, 100]
    c = [100, 100, 100, 100, 100, 130, 160, 190, 220]
    price_data = {"A": _make_df(a), "B": _make_df(b), "C": _make_df(c)}
    cfg = MomentumConfig(lookback_bars=4, rebalance_bars=4, top_k=1, atr_period=2, stop_loss_atr_mult=5.0)

    result = run_momentum_backtest(price_data, _risk_cfg(), cfg)

    rebalanced_out = [t for t in result.trades if t.exit_reason == "rebalance" and t.instrument == "A"]
    assert len(rebalanced_out) == 1


def test_result_is_usable_summary_and_return(synthetic_ohlcv):
    price_data = {"AAA": synthetic_ohlcv, "BBB": synthetic_ohlcv * 1.01}
    cfg = MomentumConfig(lookback_bars=50, rebalance_bars=50, top_k=1, atr_period=14, stop_loss_atr_mult=2.0)

    result = run_momentum_backtest(price_data, _risk_cfg(fee_pct=0.0002), cfg)

    assert isinstance(result.total_return_pct, float)
    assert isinstance(result.summary(), str)

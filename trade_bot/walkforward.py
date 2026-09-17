from __future__ import annotations

import itertools
from dataclasses import dataclass, replace

import pandas as pd

from trade_bot.backtest import BacktestResult, run_backtest
from trade_bot.config import RiskConfig, StrategyConfig
from trade_bot.strategy import generate_signals

# Parameter grid searched on each in-sample (train) window. Kept moderate in
# size since it's re-run once per window.
#
# trend_ma is deliberately NOT a grid dimension: an earlier experiment let it
# vary (None/100/150/200) alongside the other params and out-of-sample
# results got *worse* and noisier (MSFT compounded +4.90% vs. +11.33% with
# trend_ma fixed at 200 for every window; AAPL flip-flopped between "off" and
# every tested period almost window to window with no stable pattern). More
# free parameters == more ways to overfit, even for a parameter motivated by
# sound trading logic ("trade with the dominant trend"). trend_ma is kept as
# a fixed structural choice on StrategyConfig instead - set once in
# config.yaml, not re-fit per window.
DEFAULT_PARAM_GRID: list[dict] = [
    {"fast_ma": fast, "slow_ma": slow, "rsi_buy_max": rsi_buy,
     "stop_loss_atr_mult": sl_mult, "take_profit_atr_mult": tp_mult}
    for (fast, slow), rsi_buy, sl_mult, tp_mult in itertools.product(
        [(10, 30), (10, 50), (20, 50), (20, 100)],
        [55, 65, 75],
        [1.5, 2.0, 3.0],
        [3.0, 4.0, 6.0],
    )
]


@dataclass
class WindowResult:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    strategy_cfg: StrategyConfig
    risk_cfg: RiskConfig
    train_result: BacktestResult
    test_result: BacktestResult


def _best_on_window(
    df: pd.DataFrame,
    base_strategy: StrategyConfig,
    base_risk: RiskConfig,
    param_grid: list[dict],
    min_trades: int,
) -> tuple[StrategyConfig, RiskConfig, BacktestResult] | None:
    best: tuple[StrategyConfig, RiskConfig, BacktestResult] | None = None
    for params in param_grid:
        strategy_fields = {k: v for k, v in params.items() if k in StrategyConfig.__dataclass_fields__}
        risk_fields = {k: v for k, v in params.items() if k in RiskConfig.__dataclass_fields__}
        strategy_cfg = replace(base_strategy, **strategy_fields)
        risk_cfg = replace(base_risk, **risk_fields)

        signals = generate_signals(df, strategy_cfg)
        result = run_backtest(signals, risk_cfg)
        closed = [t for t in result.trades if t.exit_price is not None]
        if len(closed) < min_trades:
            continue

        if best is None or result.total_return_pct > best[2].total_return_pct:
            best = (strategy_cfg, risk_cfg, result)
    return best


def run_walk_forward(
    df: pd.DataFrame,
    base_strategy: StrategyConfig,
    base_risk: RiskConfig,
    train_bars: int,
    test_bars: int,
    step_bars: int,
    param_grid: list[dict] | None = None,
    min_trades: int = 5,
) -> list[WindowResult]:
    """Rolling walk-forward validation: on each window, pick the best
    parameters on the train slice (in-sample), then evaluate those exact
    parameters - unmodified - on the following test slice (out-of-sample).

    This is the honest way to judge a strategy: in-sample numbers from
    ad-hoc grid search are always optimistic (overfit to that one window).
    Chaining many out-of-sample test windows together gives a realistic
    read on whether the edge holds up on unseen data.
    """
    grid = param_grid if param_grid is not None else DEFAULT_PARAM_GRID

    windows: list[WindowResult] = []
    start = 0
    n = len(df)
    while start + train_bars + test_bars <= n:
        train_df = df.iloc[start : start + train_bars]
        test_start_idx = start + train_bars
        test_df = df.iloc[test_start_idx : test_start_idx + test_bars]

        best = _best_on_window(train_df, base_strategy, base_risk, grid, min_trades)
        if best is not None:
            strategy_cfg, risk_cfg, train_result = best

            # Feed indicators enough lookback from before the test window so
            # they're warm at the test window's first bar, then slice back
            # down to just the test window for the actual evaluation. Based
            # on this window's chosen params (trend_ma can vary per window).
            warmup_bars = max(strategy_cfg.slow_ma, strategy_cfg.rsi_period, strategy_cfg.atr_period) * 3
            if strategy_cfg.trend_ma:
                warmup_bars = max(warmup_bars, strategy_cfg.trend_ma * 3)
            eval_df = df.iloc[max(0, test_start_idx - warmup_bars) : test_start_idx + test_bars]
            eval_signals = generate_signals(eval_df, strategy_cfg)
            eval_signals = eval_signals.loc[test_df.index[0] :]
            test_result = run_backtest(eval_signals, risk_cfg)

            windows.append(
                WindowResult(
                    train_start=train_df.index[0],
                    train_end=train_df.index[-1],
                    test_start=test_df.index[0],
                    test_end=test_df.index[-1],
                    strategy_cfg=strategy_cfg,
                    risk_cfg=risk_cfg,
                    train_result=train_result,
                    test_result=test_result,
                )
            )

        start += step_bars

    return windows


def compounded_out_of_sample_return_pct(windows: list[WindowResult]) -> float:
    """Chains each window's out-of-sample return as if trading through them
    back to back (compounding), which is what actually matters: does the
    edge survive when parameters are re-picked on stale, non-overlapping
    data each time.
    """
    equity = 1.0
    for w in windows:
        equity *= 1 + w.test_result.total_return_pct / 100
    return (equity - 1) * 100

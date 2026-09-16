from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from trade_bot.config import RiskConfig
from trade_bot.risk import size_position


@dataclass
class Trade:
    entry_time: pd.Timestamp
    entry_price: float
    exit_time: pd.Timestamp | None = None
    exit_price: float | None = None
    quantity: float = 0.0
    exit_reason: str | None = None
    pnl: float = 0.0


@dataclass
class BacktestResult:
    trades: list[Trade] = field(default_factory=list)
    equity_curve: pd.Series = field(default_factory=pd.Series)

    @property
    def final_equity(self) -> float:
        return float(self.equity_curve.iloc[-1]) if len(self.equity_curve) else 0.0

    @property
    def total_return_pct(self) -> float:
        if len(self.equity_curve) < 2:
            return 0.0
        start = self.equity_curve.iloc[0]
        return (self.final_equity / start - 1) * 100

    @property
    def win_rate_pct(self) -> float:
        closed = [t for t in self.trades if t.exit_price is not None]
        if not closed:
            return 0.0
        wins = sum(1 for t in closed if t.pnl > 0)
        return wins / len(closed) * 100

    @property
    def max_drawdown_pct(self) -> float:
        if len(self.equity_curve) < 2:
            return 0.0
        running_max = self.equity_curve.cummax()
        drawdown = (self.equity_curve - running_max) / running_max
        return float(drawdown.min() * 100)

    def summary(self) -> str:
        closed = [t for t in self.trades if t.exit_price is not None]
        return (
            f"Trades: {len(closed)} | Win rate: {self.win_rate_pct:.1f}% | "
            f"Total return: {self.total_return_pct:.2f}% | "
            f"Max drawdown: {self.max_drawdown_pct:.2f}% | "
            f"Final equity: {self.final_equity:.2f}"
        )


def run_backtest(df: pd.DataFrame, risk_cfg: RiskConfig) -> BacktestResult:
    """Event-driven single-position backtest over a signals DataFrame
    (as produced by strategy.generate_signals).

    Each bar: if in a position, check stop-loss/take-profit against the
    bar's high/low first, then an explicit exit_signal. If flat, check
    entry_signal.
    """
    equity = risk_cfg.initial_capital
    equity_curve: dict[pd.Timestamp, float] = {}
    trades: list[Trade] = []
    open_trade: Trade | None = None
    stop_loss_price = 0.0
    take_profit_price = 0.0

    for ts, row in df.iterrows():
        if open_trade is not None:
            exit_price = None
            exit_reason = None

            if row["low"] <= stop_loss_price:
                exit_price = stop_loss_price
                exit_reason = "stop_loss"
            elif row["high"] >= take_profit_price:
                exit_price = take_profit_price
                exit_reason = "take_profit"
            elif row["exit_signal"]:
                exit_price = row["close"]
                exit_reason = "signal"

            if exit_price is not None:
                gross = (exit_price - open_trade.entry_price) * open_trade.quantity
                fee = exit_price * open_trade.quantity * risk_cfg.fee_pct
                pnl = gross - fee

                open_trade.exit_time = ts
                open_trade.exit_price = exit_price
                open_trade.exit_reason = exit_reason
                open_trade.pnl = pnl

                equity += pnl
                trades.append(open_trade)
                open_trade = None

        elif row["entry_signal"]:
            sizing = size_position(row["close"], equity, risk_cfg)
            if sizing.quantity > 0:
                entry_fee = row["close"] * sizing.quantity * risk_cfg.fee_pct
                equity -= entry_fee

                open_trade = Trade(
                    entry_time=ts,
                    entry_price=row["close"],
                    quantity=sizing.quantity,
                )
                stop_loss_price = sizing.stop_loss_price
                take_profit_price = sizing.take_profit_price

        mark_to_market = equity
        if open_trade is not None:
            mark_to_market += (row["close"] - open_trade.entry_price) * open_trade.quantity
        equity_curve[ts] = mark_to_market

    if open_trade is not None:
        trades.append(open_trade)  # still open at end of data, recorded without exit

    return BacktestResult(trades=trades, equity_curve=pd.Series(equity_curve))

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from trade_bot.config import RiskConfig
from trade_bot.strategy import atr as compute_atr


@dataclass
class MomentumConfig:
    lookback_bars: int = 420   # ~3 months of H1 bars (~7 bars/trading day * ~60 trading days)
    rebalance_bars: int = 140  # ~1 month of H1 bars - how often to re-rank and reallocate
    top_k: int = 3             # hold the top-K positive-momentum instruments, equal-weighted
    atr_period: int = 14
    stop_loss_atr_mult: float = 2.0


@dataclass
class MomentumTrade:
    instrument: str
    entry_time: pd.Timestamp
    entry_price: float
    quantity: float
    exit_time: pd.Timestamp | None = None
    exit_price: float | None = None
    exit_reason: str | None = None
    pnl: float = 0.0


@dataclass
class MomentumBacktestResult:
    trades: list[MomentumTrade] = field(default_factory=list)
    equity_curve: pd.Series = field(default_factory=pd.Series)

    @property
    def final_equity(self) -> float:
        return float(self.equity_curve.iloc[-1]) if len(self.equity_curve) else 0.0

    @property
    def total_return_pct(self) -> float:
        if len(self.equity_curve) < 2:
            return 0.0
        return (self.final_equity / self.equity_curve.iloc[0] - 1) * 100

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


def _rank_top_momentum(
    closes_now: dict[str, float], closes_lookback: dict[str, float], top_k: int
) -> list[str]:
    """Ranks instruments by trailing return (now vs. `lookback_bars` ago),
    keeping only POSITIVE momentum names - never forces a long position into
    an instrument that's already trending down, even if it's the "best of a
    bad bunch". Returns at most `top_k` instrument names, best first.
    """
    momentum = {
        inst: closes_now[inst] / closes_lookback[inst] - 1
        for inst in closes_now
        if inst in closes_lookback and closes_lookback[inst] > 0
    }
    positive = {inst: m for inst, m in momentum.items() if m > 0}
    ranked = sorted(positive, key=lambda inst: positive[inst], reverse=True)
    return ranked[:top_k]


def run_momentum_backtest(
    price_data: dict[str, pd.DataFrame], risk_cfg: RiskConfig, cfg: MomentumConfig
) -> MomentumBacktestResult:
    """Cross-sectional relative-momentum backtest across a universe of
    instruments at once - unlike the other strategies, momentum only makes
    sense ranked against a universe (a single instrument has nothing to be
    "relatively" strong against), so this can't reuse the single-instrument
    `backtest.run_backtest` loop.

    Every `rebalance_bars`, ranks all instruments by trailing return over
    the past `lookback_bars` and holds the top `top_k` instruments with
    POSITIVE momentum, equal-weighted by capital at the time each position
    is opened (existing winners that stay in the top-K are left running,
    not resized, to avoid needless turnover/fees). Between rebalances, an
    ATR-based stop-loss can close a position early.
    """
    common_index = None
    for df in price_data.values():
        common_index = df.index if common_index is None else common_index.intersection(df.index)
    common_index = common_index.sort_values()

    aligned = {inst: df.loc[common_index] for inst, df in price_data.items()}
    atrs = {inst: compute_atr(df, cfg.atr_period) for inst, df in aligned.items()}

    equity = risk_cfg.initial_capital
    equity_curve: dict[pd.Timestamp, float] = {}
    trades: list[MomentumTrade] = []
    open_positions: dict[str, MomentumTrade] = {}
    stop_prices: dict[str, float] = {}

    for i, ts in enumerate(common_index):
        for inst in list(open_positions):
            low = aligned[inst]["low"].iloc[i]
            if low <= stop_prices[inst]:
                trade = open_positions.pop(inst)
                exit_price = stop_prices.pop(inst)
                fee = exit_price * trade.quantity * risk_cfg.fee_pct
                trade.exit_time, trade.exit_price, trade.exit_reason = ts, exit_price, "stop_loss"
                trade.pnl = (exit_price - trade.entry_price) * trade.quantity - fee
                equity += trade.pnl
                trades.append(trade)

        if i >= cfg.lookback_bars and (i - cfg.lookback_bars) % cfg.rebalance_bars == 0:
            lookback_idx = i - cfg.lookback_bars
            closes_now = {inst: df["close"].iloc[i] for inst, df in aligned.items()}
            closes_then = {inst: df["close"].iloc[lookback_idx] for inst, df in aligned.items()}
            target = set(_rank_top_momentum(closes_now, closes_then, cfg.top_k))

            for inst in list(open_positions):
                if inst not in target:
                    trade = open_positions.pop(inst)
                    stop_prices.pop(inst, None)
                    exit_price = closes_now[inst]
                    fee = exit_price * trade.quantity * risk_cfg.fee_pct
                    trade.exit_time, trade.exit_price, trade.exit_reason = ts, exit_price, "rebalance"
                    trade.pnl = (exit_price - trade.entry_price) * trade.quantity - fee
                    equity += trade.pnl
                    trades.append(trade)

            new_entries = target - set(open_positions)
            if new_entries:
                allocation_per_instrument = equity / len(target)
                for inst in new_entries:
                    price = closes_now[inst]
                    atr_value = atrs[inst].iloc[i]
                    if pd.isna(atr_value) or atr_value <= 0 or price <= 0:
                        continue
                    quantity = allocation_per_instrument / price
                    if quantity <= 0:
                        continue
                    entry_fee = price * quantity * risk_cfg.fee_pct
                    equity -= entry_fee
                    open_positions[inst] = MomentumTrade(
                        instrument=inst, entry_time=ts, entry_price=price, quantity=quantity,
                    )
                    stop_prices[inst] = price - atr_value * cfg.stop_loss_atr_mult

        mark_to_market = equity
        for inst, trade in open_positions.items():
            mark_to_market += (aligned[inst]["close"].iloc[i] - trade.entry_price) * trade.quantity
        equity_curve[ts] = mark_to_market

    trades.extend(open_positions.values())  # still open at end of data, recorded without exit

    return MomentumBacktestResult(trades=trades, equity_curve=pd.Series(equity_curve))

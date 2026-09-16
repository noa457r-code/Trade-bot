from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from trade_bot.config import Config
from trade_bot.data import fetch_ohlcv, make_client
from trade_bot.risk import size_position
from trade_bot.strategy import generate_signals

logger = logging.getLogger("paper_trader")


@dataclass
class PaperPosition:
    entry_price: float
    quantity: float
    stop_loss_price: float
    take_profit_price: float


class PaperTrader:
    """Simulates trading on live market data without placing real orders.

    Safe to run continuously: only an Alpaca API key/secret pair is needed
    (market data access works the same on paper and live accounts), and no
    real orders or funds are ever involved. Note: stock market data only
    updates during exchange hours.
    """

    def __init__(self, cfg: Config, api_key: str, secret_key: str):
        self.cfg = cfg
        self.client = make_client(api_key, secret_key)
        self.equity = cfg.risk.initial_capital
        self.position: PaperPosition | None = None
        self.closed_trades: list[dict] = []

    def _fetch_recent(self):
        lookback_bars = max(self.cfg.strategy.slow_ma, self.cfg.strategy.rsi_period) * 3
        return fetch_ohlcv(
            self.client,
            self.cfg.instrument,
            self.cfg.granularity,
            max_bars=lookback_bars,
        )

    def step(self) -> None:
        df = self._fetch_recent()
        signals = generate_signals(df, self.cfg.strategy)
        last = signals.iloc[-1]
        price = float(last["close"])

        if self.position is not None:
            exit_price = None
            reason = None
            if price <= self.position.stop_loss_price:
                exit_price, reason = self.position.stop_loss_price, "stop_loss"
            elif price >= self.position.take_profit_price:
                exit_price, reason = self.position.take_profit_price, "take_profit"
            elif bool(last["exit_signal"]):
                exit_price, reason = price, "signal"

            if exit_price is not None:
                gross = (exit_price - self.position.entry_price) * self.position.quantity
                fee = exit_price * self.position.quantity * self.cfg.risk.fee_pct
                pnl = gross - fee
                self.equity += pnl
                self.closed_trades.append(
                    {
                        "entry_price": self.position.entry_price,
                        "exit_price": exit_price,
                        "quantity": self.position.quantity,
                        "reason": reason,
                        "pnl": pnl,
                    }
                )
                logger.info("EXIT (%s) @ %.5f | pnl=%.2f | equity=%.2f", reason, exit_price, pnl, self.equity)
                self.position = None

        elif bool(last["entry_signal"]):
            sizing = size_position(price, self.equity, self.cfg.risk)
            if sizing.quantity > 0:
                fee = price * sizing.quantity * self.cfg.risk.fee_pct
                self.equity -= fee
                self.position = PaperPosition(
                    entry_price=price,
                    quantity=sizing.quantity,
                    stop_loss_price=sizing.stop_loss_price,
                    take_profit_price=sizing.take_profit_price,
                )
                logger.info(
                    "ENTRY @ %.5f | qty=%.2f | stop=%.5f | target=%.5f",
                    price,
                    sizing.quantity,
                    sizing.stop_loss_price,
                    sizing.take_profit_price,
                )
        else:
            logger.debug("No signal @ %.5f | equity=%.2f", price, self.equity)

    def run_forever(self) -> None:
        logger.info(
            "Starting paper trading on %s %s (no real funds are used)", self.cfg.instrument, self.cfg.granularity
        )
        while True:
            try:
                self.step()
            except Exception:
                logger.exception("Error during paper trading step")
            time.sleep(self.cfg.paper_trading.poll_interval_seconds)

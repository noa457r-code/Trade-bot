from __future__ import annotations

import logging
import math
import time

from alpaca.common.exceptions import APIError
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderClass, OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest, StopLossRequest, TakeProfitRequest

from trade_bot.config import Config
from trade_bot.data import fetch_ohlcv, make_client
from trade_bot.risk import size_position
from trade_bot.strategy import generate_signals

logger = logging.getLogger("paper_trader")


class PaperTrader:
    """Trades on Alpaca's paper account using live market data.

    Entries are submitted as bracket orders (market entry + stop-loss +
    take-profit legs), so Alpaca handles exits at those levels itself. A
    strategy exit signal closes the position early via `close_position`.
    Always targets the `paper` trading endpoint - no real funds are ever
    involved.
    """

    def __init__(self, cfg: Config, api_key: str, secret_key: str):
        self.cfg = cfg
        self.data_client = make_client(api_key, secret_key)
        self.trading_client = TradingClient(api_key, secret_key, paper=True)

    def _fetch_recent(self):
        lookback_bars = max(self.cfg.strategy.slow_ma, self.cfg.strategy.rsi_period) * 3
        return fetch_ohlcv(
            self.data_client,
            self.cfg.instrument,
            self.cfg.granularity,
            max_bars=lookback_bars,
        )

    def _current_position(self):
        try:
            return self.trading_client.get_open_position(self.cfg.instrument)
        except APIError:
            return None

    def step(self) -> None:
        df = self._fetch_recent()
        signals = generate_signals(df, self.cfg.strategy)
        last = signals.iloc[-1]
        price = float(last["close"])

        position = self._current_position()

        if position is not None:
            if bool(last["exit_signal"]):
                self.trading_client.close_position(self.cfg.instrument)
                logger.info("EXIT (signal) @ ~%.2f | qty=%s", price, position.qty)
            else:
                logger.debug("Holding qty=%s @ ~%.2f", position.qty, price)
            return

        if not bool(last["entry_signal"]):
            logger.debug("No signal @ %.2f", price)
            return

        equity = float(self.trading_client.get_account().equity)
        sizing = size_position(price, float(last["atr"]), equity, self.cfg.risk)
        quantity = math.floor(sizing.quantity)
        if quantity < 1:
            logger.debug("Signal but position size < 1 share @ %.2f | equity=%.2f", price, equity)
            return

        order = MarketOrderRequest(
            symbol=self.cfg.instrument,
            qty=quantity,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY,
            order_class=OrderClass.BRACKET,
            take_profit=TakeProfitRequest(limit_price=round(sizing.take_profit_price, 2)),
            stop_loss=StopLossRequest(stop_price=round(sizing.stop_loss_price, 2)),
        )
        self.trading_client.submit_order(order)
        logger.info(
            "ENTRY @ %.2f | qty=%d | stop=%.2f | target=%.2f",
            price, quantity, sizing.stop_loss_price, sizing.take_profit_price,
        )

    def run_forever(self) -> None:
        logger.info(
            "Starting paper trading on %s %s (Alpaca paper account, no real funds)",
            self.cfg.instrument, self.cfg.granularity,
        )
        while True:
            try:
                self.step()
            except Exception:
                logger.exception("Error during paper trading step")
            time.sleep(self.cfg.paper_trading.poll_interval_seconds)

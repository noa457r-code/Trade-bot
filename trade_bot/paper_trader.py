from __future__ import annotations

import logging
import math
import time

from alpaca.common.exceptions import APIError
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderClass, OrderSide, TimeInForce
from alpaca.trading.requests import GetPortfolioHistoryRequest, MarketOrderRequest, StopLossRequest, TakeProfitRequest

from trade_bot.advisor import equity_based_tips, error_based_tips
from trade_bot.config import Config
from trade_bot.data import fetch_ohlcv, make_client
from trade_bot.notify import send_discord_notification
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

    def __init__(self, cfg: Config, api_key: str, secret_key: str, discord_webhook_url: str | None = None):
        self.cfg = cfg
        self.data_client = make_client(api_key, secret_key)
        self.trading_client = TradingClient(api_key, secret_key, paper=True)
        self.discord_webhook_url = discord_webhook_url
        self._error_streaks: dict[str, int] = {}

    def _fetch_recent(self, instrument: str):
        lookback_bars = max(self.cfg.strategy.slow_ma, self.cfg.strategy.rsi_period) * 3
        return fetch_ohlcv(
            self.data_client,
            instrument,
            self.cfg.granularity,
            max_bars=lookback_bars,
        )

    def _current_position(self, instrument: str):
        try:
            return self.trading_client.get_open_position(instrument)
        except APIError:
            return None

    def step_instrument(self, instrument: str) -> str | None:
        """Runs one strategy step for `instrument`, returns a one-line status
        string ONLY if there's something worth reporting in the periodic
        digest (an open position, or an error) - flat/no-signal instruments
        return None so the digest stays focused on what matters.
        """
        df = self._fetch_recent(instrument)
        signals = generate_signals(df, self.cfg.strategy)
        last = signals.iloc[-1]
        price = float(last["close"])

        position = self._current_position(instrument)

        if position is not None:
            if bool(last["exit_signal"]):
                self.trading_client.close_position(instrument)
                logger.info("[%s] EXIT (signal) @ ~%.2f | qty=%s", instrument, price, position.qty)
                send_discord_notification(
                    f"EXIT {instrument} @ ~{price:.2f} | qty={position.qty}",
                    self.discord_webhook_url,
                )
                return f"{instrument}: EXIT gerade ausgeloest @ {price:.2f}"
            logger.debug("[%s] Holding qty=%s @ ~%.2f", instrument, position.qty, price)
            return f"{instrument}: haelt qty={position.qty} @ {price:.2f}"

        if not bool(last["entry_signal"]):
            logger.debug("[%s] No signal @ %.2f", instrument, price)
            return None

        # Equity is re-read live before every entry, so risk_per_trade is
        # sized against current account equity across all instruments
        # sharing this one paper account.
        equity = float(self.trading_client.get_account().equity)
        sizing = size_position(price, float(last["atr"]), equity, self.cfg.risk)
        quantity = math.floor(sizing.quantity)
        if quantity < 1:
            logger.debug("[%s] Signal but position size < 1 share @ %.2f | equity=%.2f", instrument, price, equity)
            return None

        order = MarketOrderRequest(
            symbol=instrument,
            qty=quantity,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY,
            order_class=OrderClass.BRACKET,
            take_profit=TakeProfitRequest(limit_price=round(sizing.take_profit_price, 2)),
            stop_loss=StopLossRequest(stop_price=round(sizing.stop_loss_price, 2)),
        )
        self.trading_client.submit_order(order)
        logger.info(
            "[%s] ENTRY @ %.2f | qty=%d | stop=%.2f | target=%.2f",
            instrument, price, quantity, sizing.stop_loss_price, sizing.take_profit_price,
        )
        send_discord_notification(
            f"ENTRY {instrument} @ ~{price:.2f} | qty={quantity} | "
            f"stop={sizing.stop_loss_price:.2f} | target={sizing.take_profit_price:.2f}",
            self.discord_webhook_url,
        )
        return f"{instrument}: ENTRY gerade ausgeloest @ {price:.2f}"

    def step(self) -> list[str]:
        """Runs a step for every instrument, returns only the noteworthy
        status lines (open positions, entries/exits, errors) - flat/no-signal
        instruments are omitted so the periodic digest stays short.
        """
        status_lines = []
        for instrument in self.cfg.instruments:
            try:
                line = self.step_instrument(instrument)
                if line is not None:
                    status_lines.append(line)
                self._error_streaks[instrument] = 0
            except Exception:
                logger.exception("[%s] Error during paper trading step", instrument)
                status_lines.append(f"{instrument}: FEHLER beim Abfragen (siehe Log)")
                self._error_streaks[instrument] = self._error_streaks.get(instrument, 0) + 1
        return status_lines

    def _fetch_tips(self) -> list[str]:
        """Rule-based tips from real account state - only returns something
        when a threshold actually triggers (see trade_bot.advisor)."""
        tips = error_based_tips(self._error_streaks)
        try:
            history = self.trading_client.get_portfolio_history(
                GetPortfolioHistoryRequest(period="1M", timeframe="1D")
            )
            tips += equity_based_tips(list(history.equity), history.base_value)
        except Exception:
            logger.exception("Failed to fetch portfolio history for tips")
        return tips

    def send_status_digest(self, status_lines: list[str]) -> None:
        equity = float(self.trading_client.get_account().equity)
        body = "\n".join(status_lines) if status_lines else "keine offenen Positionen, keine Signale"
        message = f"Status-Update | Kontostand: {equity:.2f}\n{body}"

        tips = self._fetch_tips()
        if tips:
            message += "\n\nTipps:\n" + "\n".join(f"- {t}" for t in tips)

        send_discord_notification(message, self.discord_webhook_url)
        logger.info("Sent periodic status digest to Discord")

    def run_forever(self) -> None:
        logger.info(
            "Starting paper trading on %s %s (Alpaca paper account, no real funds)",
            ", ".join(self.cfg.instruments), self.cfg.granularity,
        )
        status_interval = self.cfg.paper_trading.status_update_minutes * 60
        last_status_at = time.monotonic()
        while True:
            status_lines = self.step()

            now = time.monotonic()
            if status_interval > 0 and now - last_status_at >= status_interval:
                try:
                    self.send_status_digest(status_lines)
                except Exception:
                    logger.exception("Failed to send periodic status digest")
                last_status_at = now

            time.sleep(self.cfg.paper_trading.poll_interval_seconds)

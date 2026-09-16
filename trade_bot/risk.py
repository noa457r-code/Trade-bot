from __future__ import annotations

from dataclasses import dataclass

from trade_bot.config import RiskConfig


@dataclass
class PositionSizing:
    quantity: float
    stop_loss_price: float
    take_profit_price: float


def size_position(entry_price: float, atr_value: float, equity: float, cfg: RiskConfig) -> PositionSizing:
    """Size a position so that hitting the stop-loss loses at most
    `risk_per_trade` of current equity.

    Stop-loss/take-profit distances scale with `atr_value` (the
    instrument's current Average True Range) instead of a fixed
    percentage, so they adapt to how much the instrument actually moves.
    """
    stop_loss_price = entry_price - atr_value * cfg.stop_loss_atr_mult
    take_profit_price = entry_price + atr_value * cfg.take_profit_atr_mult

    risk_amount = equity * cfg.risk_per_trade
    price_risk_per_unit = entry_price - stop_loss_price
    if price_risk_per_unit <= 0:
        quantity = 0.0
    else:
        quantity = risk_amount / price_risk_per_unit

    # Never risk more notional than the available equity allows.
    max_affordable_qty = equity / entry_price
    quantity = min(quantity, max_affordable_qty)

    return PositionSizing(
        quantity=quantity,
        stop_loss_price=stop_loss_price,
        take_profit_price=take_profit_price,
    )

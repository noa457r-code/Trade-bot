from trade_bot.config import RiskConfig
from trade_bot.risk import size_position


def make_risk_cfg() -> RiskConfig:
    return RiskConfig(
        initial_capital=10000,
        risk_per_trade=0.01,
        stop_loss_pct=0.02,
        take_profit_pct=0.04,
        fee_pct=0.001,
    )


def test_size_position_risks_configured_fraction():
    cfg = make_risk_cfg()
    equity = 10000
    entry_price = 100.0

    sizing = size_position(entry_price, equity, cfg)

    risk_amount = (entry_price - sizing.stop_loss_price) * sizing.quantity
    expected_risk = equity * cfg.risk_per_trade
    assert abs(risk_amount - expected_risk) < 1e-6


def test_size_position_never_exceeds_available_equity():
    cfg = make_risk_cfg()
    equity = 100  # small equity relative to risk_per_trade math
    entry_price = 50000.0  # e.g. BTC price much larger than equity

    sizing = size_position(entry_price, equity, cfg)

    assert sizing.quantity * entry_price <= equity + 1e-6

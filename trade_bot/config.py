from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class StrategyConfig:
    fast_ma: int
    slow_ma: int
    rsi_period: int
    rsi_buy_max: float
    rsi_sell_min: float
    atr_period: int
    trend_ma: int | None = None  # long-term MA; entries only taken above it (uptrend filter). None disables it.


@dataclass
class RiskConfig:
    initial_capital: float
    risk_per_trade: float
    stop_loss_atr_mult: float
    take_profit_atr_mult: float
    fee_pct: float
    max_open_positions: int = 999  # hard cap on simultaneously open positions across all instruments
    kill_switch_drawdown_pct: float = 10.0  # bot stops trading + closes positions past this drawdown from peak equity
    breakeven_trigger_atr_mult: float = 1.0  # once unrealized profit reaches this many ATR, stop moves to entry price


@dataclass
class PaperTradingConfig:
    poll_interval_seconds: int
    status_update_minutes: int = 60  # periodic Discord status digest, independent of poll interval. 0 disables it.


@dataclass
class Config:
    broker: str
    environment: str
    instruments: list[str]
    granularity: str
    strategy: StrategyConfig
    risk: RiskConfig
    paper_trading: PaperTradingConfig

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        with open(path, "r") as f:
            raw = yaml.safe_load(f)

        if "instruments" in raw:
            instruments = list(raw["instruments"])
        else:
            # Backward-compat: single `instrument: TICKER` key.
            instruments = [raw["instrument"]]

        return cls(
            broker=raw["broker"],
            environment=raw["environment"],
            instruments=instruments,
            granularity=raw["granularity"],
            strategy=StrategyConfig(**raw["strategy"]),
            risk=RiskConfig(**raw["risk"]),
            paper_trading=PaperTradingConfig(**raw["paper_trading"]),
        )

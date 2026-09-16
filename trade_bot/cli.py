from __future__ import annotations

import argparse
import logging

from trade_bot.backtest import run_backtest
from trade_bot.config import Config
from trade_bot.data import fetch_ohlcv, make_exchange
from trade_bot.paper_trader import PaperTrader
from trade_bot.strategy import generate_signals


def cmd_backtest(args: argparse.Namespace) -> None:
    cfg = Config.from_yaml(args.config)
    exchange = make_exchange(cfg.exchange)

    since_ms = exchange.parse8601(f"{args.since}T00:00:00Z") if args.since else None
    df = fetch_ohlcv(exchange, cfg.symbol, cfg.timeframe, since_ms=since_ms, max_bars=args.bars)

    signals = generate_signals(df, cfg.strategy)
    result = run_backtest(signals, cfg.risk)

    print(f"Symbol: {cfg.symbol} | Timeframe: {cfg.timeframe} | Bars: {len(df)}")
    print(result.summary())

    if args.trades:
        for t in result.trades:
            exit_price = f"{t.exit_price:.2f}" if t.exit_price is not None else "OPEN"
            print(
                f"  {t.entry_time} entry={t.entry_price:.2f} -> "
                f"{t.exit_time} exit={exit_price} ({t.exit_reason}) pnl={t.pnl:.2f}"
            )


def cmd_paper(args: argparse.Namespace) -> None:
    cfg = Config.from_yaml(args.config)
    trader = PaperTrader(cfg)
    trader.run_forever()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(prog="trade-bot")
    subparsers = parser.add_subparsers(dest="command", required=True)

    backtest_parser = subparsers.add_parser("backtest", help="Run a historical backtest")
    backtest_parser.add_argument("--config", default="config.yaml")
    backtest_parser.add_argument("--since", default=None, help="ISO date, e.g. 2023-01-01")
    backtest_parser.add_argument("--bars", type=int, default=2000)
    backtest_parser.add_argument("--trades", action="store_true", help="Print individual trades")
    backtest_parser.set_defaults(func=cmd_backtest)

    paper_parser = subparsers.add_parser("paper", help="Run continuous paper trading (no real funds)")
    paper_parser.add_argument("--config", default="config.yaml")
    paper_parser.set_defaults(func=cmd_paper)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

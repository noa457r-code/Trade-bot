from __future__ import annotations

import argparse
import logging
import os
from datetime import datetime, timezone

from dotenv import load_dotenv

from trade_bot.backtest import run_backtest
from trade_bot.config import Config
from trade_bot.data import fetch_ohlcv, make_client
from trade_bot.paper_trader import PaperTrader
from trade_bot.strategy import generate_signals
from trade_bot.walkforward import compounded_out_of_sample_return_pct, run_walk_forward


def _require_alpaca_credentials() -> tuple[str, str]:
    api_key = os.environ.get("ALPACA_API_KEY")
    secret_key = os.environ.get("ALPACA_SECRET_KEY")
    if not api_key or not secret_key:
        raise SystemExit(
            "ALPACA_API_KEY / ALPACA_SECRET_KEY sind nicht gesetzt. Kopiere .env.example "
            "nach .env und trage deine (kostenlosen) Alpaca-Paper-Account-Keys ein."
        )
    return api_key, secret_key


def cmd_backtest(args: argparse.Namespace) -> None:
    load_dotenv()
    cfg = Config.from_yaml(args.config)
    api_key, secret_key = _require_alpaca_credentials()
    client = make_client(api_key, secret_key)

    since = None
    if args.since:
        since = datetime.strptime(args.since, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    # Each instrument is backtested independently against the same starting
    # capital (not a shared portfolio simulation) - this shows per-instrument
    # edge, not combined portfolio equity.
    returns = []
    for instrument in cfg.instruments:
        df = fetch_ohlcv(client, instrument, cfg.granularity, since=since, max_bars=args.bars)

        signals = generate_signals(df, cfg.strategy)
        result = run_backtest(signals, cfg.risk)
        returns.append(result.total_return_pct)

        print(f"Instrument: {instrument} | Granularity: {cfg.granularity} | Bars: {len(df)}")
        print(result.summary())

        if args.trades:
            for t in result.trades:
                exit_price = f"{t.exit_price:.5f}" if t.exit_price is not None else "OPEN"
                print(
                    f"  {t.entry_time} entry={t.entry_price:.5f} -> "
                    f"{t.exit_time} exit={exit_price} ({t.exit_reason}) pnl={t.pnl:.2f}"
                )
        print()

    if len(cfg.instruments) > 1:
        avg_return = sum(returns) / len(returns)
        print(f"Average return across {len(cfg.instruments)} instruments: {avg_return:.2f}%")


def cmd_walkforward(args: argparse.Namespace) -> None:
    load_dotenv()
    cfg = Config.from_yaml(args.config)
    api_key, secret_key = _require_alpaca_credentials()
    client = make_client(api_key, secret_key)

    since = None
    if args.since:
        since = datetime.strptime(args.since, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    for instrument in cfg.instruments:
        df = fetch_ohlcv(client, instrument, cfg.granularity, since=since, max_bars=args.bars)
        windows = run_walk_forward(
            df,
            base_strategy=cfg.strategy,
            base_risk=cfg.risk,
            train_bars=args.train_bars,
            test_bars=args.test_bars,
            step_bars=args.step_bars,
        )

        print(f"\n=== {instrument} | {len(windows)} walk-forward windows ===")
        if not windows:
            print("  No window produced enough trades to evaluate - widen the date range or lower --min-trades.")
            continue

        for w in windows:
            print(
                f"  train {w.train_start.date()}..{w.train_end.date()} "
                f"(fast={w.strategy_cfg.fast_ma} slow={w.strategy_cfg.slow_ma} "
                f"trend={w.strategy_cfg.trend_ma or 'off'} "
                f"sl={w.risk_cfg.stop_loss_atr_mult}x tp={w.risk_cfg.take_profit_atr_mult}x, "
                f"in-sample {w.train_result.total_return_pct:+.2f}%) "
                f"-> test {w.test_start.date()}..{w.test_end.date()} "
                f"out-of-sample {w.test_result.total_return_pct:+.2f}% "
                f"(win {w.test_result.win_rate_pct:.1f}%, dd {w.test_result.max_drawdown_pct:.2f}%)"
            )

        compounded = compounded_out_of_sample_return_pct(windows)
        print(f"  Compounded out-of-sample return across all windows: {compounded:+.2f}%")


def cmd_paper(args: argparse.Namespace) -> None:
    load_dotenv()
    cfg = Config.from_yaml(args.config)
    api_key, secret_key = _require_alpaca_credentials()
    discord_webhook_url = os.environ.get("DISCORD_WEBHOOK_URL")
    trader = PaperTrader(cfg, api_key=api_key, secret_key=secret_key, discord_webhook_url=discord_webhook_url)
    trader.run_forever()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(prog="trade-bot")
    subparsers = parser.add_subparsers(dest="command", required=True)

    backtest_parser = subparsers.add_parser("backtest", help="Run a historical backtest")
    backtest_parser.add_argument("--config", default="config.yaml")
    backtest_parser.add_argument("--since", default=None, help="Datum, z.B. 2023-01-01")
    backtest_parser.add_argument("--bars", type=int, default=2000)
    backtest_parser.add_argument("--trades", action="store_true", help="Print individual trades")
    backtest_parser.set_defaults(func=cmd_backtest)

    paper_parser = subparsers.add_parser("paper", help="Run continuous paper trading (no real funds)")
    paper_parser.add_argument("--config", default="config.yaml")
    paper_parser.set_defaults(func=cmd_paper)

    wf_parser = subparsers.add_parser(
        "walkforward", help="Rolling walk-forward validation: refit params per window, test out-of-sample"
    )
    wf_parser.add_argument("--config", default="config.yaml")
    wf_parser.add_argument("--since", default=None, help="Datum, z.B. 2021-01-01")
    wf_parser.add_argument("--bars", type=int, default=8000)
    wf_parser.add_argument("--train-bars", type=int, default=2000, help="Bars per in-sample fitting window")
    wf_parser.add_argument("--test-bars", type=int, default=500, help="Bars per out-of-sample test window")
    wf_parser.add_argument("--step-bars", type=int, default=500, help="Bars to roll forward between windows")
    wf_parser.set_defaults(func=cmd_walkforward)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

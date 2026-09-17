from __future__ import annotations

# Simple rule-based tips derived from real account state (equity history from
# Alpaca's portfolio history, plus the current step's status lines) - not
# guesses. Only returns something when a threshold actually triggers, so the
# Discord digest doesn't get a "tip" attached every time.

DRAWDOWN_WARN_PCT = -5.0
DRAWDOWN_SEVERE_PCT = -10.0
TOTAL_RETURN_WARN_PCT = -5.0
REPEATED_ERROR_COUNT = 2  # same instrument erroring this many times in a row


def _nonzero_equity(equity_series: list[float]) -> list[float]:
    # Alpaca's portfolio history pads unfunded days with 0.0 before the
    # account existed - those aren't real equity readings.
    return [e for e in equity_series if e > 0]


def equity_based_tips(equity_series: list[float], base_value: float) -> list[str]:
    values = _nonzero_equity(equity_series)
    if len(values) < 2 or base_value <= 0:
        return []

    current = values[-1]
    peak = max(values)
    drawdown_pct = (current - peak) / peak * 100
    total_return_pct = (current / base_value - 1) * 100

    tips = []
    if drawdown_pct <= DRAWDOWN_SEVERE_PCT:
        tips.append(
            f"Drawdown erheblich bei {drawdown_pct:.1f}% vom Hoch - Pausieren und "
            f"Strategie per `walkforward` gegenpruefen erwaegen."
        )
    elif drawdown_pct <= DRAWDOWN_WARN_PCT:
        tips.append(
            f"Drawdown bei {drawdown_pct:.1f}% vom Hoch - im Blick behalten, ggf. "
            f"risk_per_trade senken."
        )

    if total_return_pct <= TOTAL_RETURN_WARN_PCT:
        tips.append(
            f"Gesamtrendite seit Start negativ ({total_return_pct:.1f}%) - "
            f"Parameter/Trendfilter per `walkforward` gegenpruefen."
        )

    return tips


def error_based_tips(instrument_error_counts: dict[str, int]) -> list[str]:
    tips = []
    for instrument, count in instrument_error_counts.items():
        if count >= REPEATED_ERROR_COUNT:
            tips.append(
                f"{instrument}: wiederholt Fehler ({count}x in Folge) - "
                f"API-Key/Netzwerk/Symbol pruefen."
            )
    return tips

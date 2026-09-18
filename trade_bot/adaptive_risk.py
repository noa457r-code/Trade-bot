from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

STREAK_THRESHOLD = 2
STEP_DOWN = 0.75
STEP_UP = 1.25
MIN_MULTIPLIER = 0.25
MAX_MULTIPLIER = 1.0


@dataclass
class AdaptiveRiskState:
    """Persisted across restarts (same reasoning as SafetyState - systemd
    auto-restarts the bot, so a crash shouldn't silently reset a hard-won
    lesson from a recent losing streak).
    """

    current_multiplier: float = 1.0
    consecutive_wins: int = 0
    consecutive_losses: int = 0

    @classmethod
    def load(cls, path: str | Path) -> "AdaptiveRiskState":
        p = Path(path)
        if not p.exists():
            return cls()
        with open(p, "r") as f:
            raw = json.load(f)
        return cls(**raw)

    def save(self, path: str | Path) -> None:
        with open(path, "w") as f:
            json.dump(
                {
                    "current_multiplier": self.current_multiplier,
                    "consecutive_wins": self.consecutive_wins,
                    "consecutive_losses": self.consecutive_losses,
                },
                f,
            )


def record_trade_outcome(state: AdaptiveRiskState, pnl: float) -> None:
    """Updates the win/loss streak from one closed trade's realized P&L. A
    breakeven trade (pnl == 0) affects neither streak.
    """
    if pnl > 0:
        state.consecutive_wins += 1
        state.consecutive_losses = 0
    elif pnl < 0:
        state.consecutive_losses += 1
        state.consecutive_wins = 0


def adjust_multiplier(state: AdaptiveRiskState) -> bool:
    """Applies one step change once a streak reaches STREAK_THRESHOLD, then
    resets that streak counter - so it takes a fresh streak to trigger
    again, not every single loss/win beyond the threshold. Returns True
    only when the multiplier actually moved (never past the floor/ceiling).
    """
    if state.consecutive_losses >= STREAK_THRESHOLD:
        new_multiplier = max(MIN_MULTIPLIER, state.current_multiplier * STEP_DOWN)
        state.consecutive_losses = 0
        changed = new_multiplier != state.current_multiplier
        state.current_multiplier = new_multiplier
        return changed

    if state.consecutive_wins >= STREAK_THRESHOLD:
        new_multiplier = min(MAX_MULTIPLIER, state.current_multiplier * STEP_UP)
        state.consecutive_wins = 0
        changed = new_multiplier != state.current_multiplier
        state.current_multiplier = new_multiplier
        return changed

    return False

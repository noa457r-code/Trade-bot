from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SafetyState:
    """Persisted across restarts (systemd auto-restarts the bot on crash) so
    a restart right after a big loss can't silently reset the drawdown peak
    or un-kill an already-tripped kill switch.
    """

    peak_equity: float = 0.0
    killed: bool = False

    @classmethod
    def load(cls, path: str | Path) -> "SafetyState":
        p = Path(path)
        if not p.exists():
            return cls()
        with open(p, "r") as f:
            raw = json.load(f)
        return cls(peak_equity=raw["peak_equity"], killed=raw["killed"])

    def save(self, path: str | Path) -> None:
        with open(path, "w") as f:
            json.dump({"peak_equity": self.peak_equity, "killed": self.killed}, f)


def check_kill_switch(equity: float, state: SafetyState, threshold_pct: float) -> bool:
    """Updates `state.peak_equity` (only upward) and flips `state.killed` to
    True the moment the drawdown from peak first crosses -threshold_pct.
    Returns True only on that first trigger - once already killed, this
    never re-triggers (a human must reset it).
    """
    if equity > state.peak_equity:
        state.peak_equity = equity

    if state.killed or state.peak_equity <= 0:
        return False

    drawdown_pct = (equity - state.peak_equity) / state.peak_equity * 100
    if drawdown_pct <= -threshold_pct:
        state.killed = True
        return True
    return False


def is_position_limit_reached(open_count: int, max_positions: int) -> bool:
    return open_count >= max_positions


def reset_kill_switch(path: str | Path) -> SafetyState:
    """Clears the killed flag so the bot resumes trading. Keeps peak_equity
    as-is - a reset shouldn't let a fresh drawdown window start from a lower
    baseline than the account actually reached.
    """
    state = SafetyState.load(path)
    state.killed = False
    state.save(path)
    return state

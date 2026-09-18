from __future__ import annotations

from trade_bot.adaptive_risk import AdaptiveRiskState, adjust_multiplier, record_trade_outcome


def test_win_increments_win_streak_and_resets_loss_streak():
    state = AdaptiveRiskState(consecutive_losses=3)
    record_trade_outcome(state, pnl=50.0)
    assert state.consecutive_wins == 1
    assert state.consecutive_losses == 0


def test_loss_increments_loss_streak_and_resets_win_streak():
    state = AdaptiveRiskState(consecutive_wins=3)
    record_trade_outcome(state, pnl=-50.0)
    assert state.consecutive_losses == 1
    assert state.consecutive_wins == 0


def test_zero_pnl_does_not_affect_streaks():
    state = AdaptiveRiskState(consecutive_wins=1, consecutive_losses=0)
    record_trade_outcome(state, pnl=0.0)
    assert state.consecutive_wins == 1
    assert state.consecutive_losses == 0


def test_no_adjustment_below_streak_threshold():
    state = AdaptiveRiskState(consecutive_losses=1)
    changed = adjust_multiplier(state)
    assert changed is False
    assert state.current_multiplier == 1.0


def test_two_losses_in_a_row_reduce_multiplier_by_25_percent():
    state = AdaptiveRiskState(consecutive_losses=2)
    changed = adjust_multiplier(state)
    assert changed is True
    assert state.current_multiplier == 0.75
    assert state.consecutive_losses == 0  # reset so it takes a fresh streak to trigger again


def test_multiplier_never_drops_below_floor():
    state = AdaptiveRiskState(current_multiplier=0.25, consecutive_losses=2)
    changed = adjust_multiplier(state)
    assert changed is False
    assert state.current_multiplier == 0.25
    assert state.consecutive_losses == 0  # still resets, just no further reduction


def test_two_wins_in_a_row_increase_multiplier_by_25_percent():
    state = AdaptiveRiskState(current_multiplier=0.75, consecutive_wins=2)
    changed = adjust_multiplier(state)
    assert changed is True
    assert round(state.current_multiplier, 4) == round(0.75 * 1.25, 4)
    assert state.consecutive_wins == 0


def test_multiplier_never_exceeds_configured_ceiling():
    state = AdaptiveRiskState(current_multiplier=1.0, consecutive_wins=2)
    changed = adjust_multiplier(state)
    assert changed is False
    assert state.current_multiplier == 1.0
    assert state.consecutive_wins == 0


def test_state_round_trips_through_save_and_load(tmp_path):
    path = tmp_path / "adaptive_risk_state.json"
    state = AdaptiveRiskState(current_multiplier=0.5625, consecutive_wins=1, consecutive_losses=0)
    state.save(path)

    loaded = AdaptiveRiskState.load(path)
    assert loaded.current_multiplier == 0.5625
    assert loaded.consecutive_wins == 1
    assert loaded.consecutive_losses == 0


def test_load_missing_file_returns_fresh_state(tmp_path):
    state = AdaptiveRiskState.load(tmp_path / "does_not_exist.json")
    assert state.current_multiplier == 1.0
    assert state.consecutive_wins == 0
    assert state.consecutive_losses == 0

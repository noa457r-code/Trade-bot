from __future__ import annotations

from trade_bot.safety import SafetyState, check_kill_switch, is_position_limit_reached, reset_kill_switch


def test_peak_equity_tracks_new_highs():
    state = SafetyState(peak_equity=100.0)
    check_kill_switch(120.0, state, threshold_pct=10)
    assert state.peak_equity == 120.0


def test_no_trigger_within_threshold():
    state = SafetyState(peak_equity=100.0)
    triggered = check_kill_switch(95.0, state, threshold_pct=10)
    assert triggered is False
    assert state.killed is False


def test_triggers_when_drawdown_exceeds_threshold():
    state = SafetyState(peak_equity=100.0)
    triggered = check_kill_switch(89.0, state, threshold_pct=10)
    assert triggered is True
    assert state.killed is True


def test_does_not_retrigger_once_already_killed():
    state = SafetyState(peak_equity=100.0, killed=True)
    triggered = check_kill_switch(50.0, state, threshold_pct=10)
    assert triggered is False


def test_state_round_trips_through_save_and_load(tmp_path):
    path = tmp_path / "safety_state.json"
    state = SafetyState(peak_equity=12345.67, killed=True)
    state.save(path)

    loaded = SafetyState.load(path)
    assert loaded.peak_equity == 12345.67
    assert loaded.killed is True


def test_load_missing_file_returns_fresh_state(tmp_path):
    state = SafetyState.load(tmp_path / "does_not_exist.json")
    assert state.peak_equity == 0.0
    assert state.killed is False


def test_position_limit_reached_at_max():
    assert is_position_limit_reached(open_count=2, max_positions=2) is True


def test_position_limit_not_reached_below_max():
    assert is_position_limit_reached(open_count=1, max_positions=2) is False


def test_reset_kill_switch_clears_killed_but_keeps_peak_equity(tmp_path):
    path = tmp_path / "safety_state.json"
    SafetyState(peak_equity=50000.0, killed=True).save(path)

    result = reset_kill_switch(path)

    assert result.killed is False
    assert result.peak_equity == 50000.0
    reloaded = SafetyState.load(path)
    assert reloaded.killed is False
    assert reloaded.peak_equity == 50000.0

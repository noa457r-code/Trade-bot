from trade_bot.advisor import equity_based_tips, error_based_tips


def test_no_tips_when_equity_flat_at_base():
    tips = equity_based_tips([0.0, 0.0, 100000.0, 100000.0], base_value=100000.0)
    assert tips == []


def test_no_tips_with_too_little_history():
    tips = equity_based_tips([100000.0], base_value=100000.0)
    assert tips == []


def test_warns_on_moderate_drawdown():
    # Peak 100000, current 93000 -> -7% drawdown, above the -5% warn threshold.
    tips = equity_based_tips([0.0, 100000.0, 93000.0], base_value=100000.0)
    assert any("Drawdown" in t for t in tips)
    assert not any("erheblich" in t for t in tips)


def test_escalates_on_severe_drawdown():
    # Peak 100000, current 85000 -> -15% drawdown, past the -10% severe threshold.
    tips = equity_based_tips([0.0, 100000.0, 85000.0], base_value=100000.0)
    assert any("erheblich" in t for t in tips)


def test_warns_on_negative_total_return():
    # Never above base_value, ends down -6% overall even without a big single drawdown.
    tips = equity_based_tips([0.0, 96000.0, 95000.0, 94000.0], base_value=100000.0)
    assert any("Gesamtrendite" in t for t in tips)


def test_error_tips_only_above_threshold():
    assert error_based_tips({"AAPL": 1}) == []
    tips = error_based_tips({"AAPL": 2, "MSFT": 0})
    assert len(tips) == 1
    assert "AAPL" in tips[0]

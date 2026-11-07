import numpy as np
import pandas as pd
import pytest

from quant_lab.engine.backtest import Signal, simulate
from quant_lab.engine.costs import ZERO_COST, CostModel
from quant_lab.metrics import max_drawdown, summarize
from quant_lab.orderflow import cvd, signed_volume


def bars_from(rows):
    idx = pd.date_range("2024-01-01", periods=len(rows), freq="15min", tz="UTC")
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx)


def test_round_trip_cost_defaults():
    c = CostModel()
    assert c.round_trip("taker", "taker") == pytest.approx(0.0013)  # 0.05%*2 + 1.5bp*2
    assert c.round_trip("maker", "maker") == pytest.approx(0.0004)


def test_market_entry_fills_next_open_and_hits_target():
    b = bars_from([[100, 101, 99, 100], [100, 103, 99.5, 102], [102, 106, 101, 105]])
    t = simulate(b, [Signal(bar=0, side=1, stop=98, target=105)], ZERO_COST)
    assert t.iloc[0]["entry_px"] == 100
    assert t.iloc[0]["exit_reason"] == "target"
    assert t.iloc[0]["gross_ret"] == pytest.approx(0.05)


def test_stop_wins_when_both_inside_same_bar():
    b = bars_from([[100, 100, 100, 100], [100, 110, 90, 100]])
    t = simulate(b, [Signal(bar=0, side=1, stop=95, target=105)], ZERO_COST)
    assert t.iloc[0]["exit_reason"] == "stop"


def test_through_fill_rule_rejects_touch_only():
    # low touches 99 exactly but never trades below it
    b = bars_from([[100, 100, 100, 100], [100, 101, 99, 100.5], [100.5, 104, 100, 103]])
    sig = Signal(bar=0, side=1, stop=97, target=103, entry="limit", limit_price=99)
    assert len(simulate(b, [sig], ZERO_COST, fill="touch")) == 1
    assert len(simulate(b, [sig], ZERO_COST, fill="through")) == 0


def test_metrics_drawdown():
    eq = pd.Series([1.0, 1.2, 0.9, 1.1])
    assert max_drawdown(eq) == pytest.approx(-0.25)
    trades = pd.DataFrame({"net_ret": [0.1, -0.05], "gross_ret": [0.11, -0.04], "cost": [0.01, 0.01]})
    s = summarize(trades)
    assert s["win_rate"] == 0.5 and s["payoff_ratio"] == pytest.approx(2.0)


def test_cvd_sign_convention():
    idx = pd.date_range("2024-01-01", periods=2, freq="1s", tz="UTC")
    tr = pd.DataFrame({"quantity": [2.0, 1.0], "is_buyer_maker": [False, True]}, index=idx)
    # first trade: buyer was taker -> +2 ; second: seller was taker -> -1
    assert list(signed_volume(tr)) == [2.0, -1.0]
    assert cvd(tr).iloc[-1] == 1.0


def test_limit_fill_bar_does_not_count_target():
    # fills at 99 and the same bar's high reaches the target: order inside the bar is unknown
    b = bars_from([[100, 100, 100, 100], [100, 102, 98.9, 99.5], [99.5, 99.8, 99.2, 99.6]])
    sig = Signal(bar=0, side=1, stop=97, target=101.5, entry="limit", limit_price=99)
    t = simulate(b, [sig], ZERO_COST, max_hold=1)
    assert t.iloc[0]["exit_reason"] != "target"


def test_entry_open_beyond_stop_exits_at_open_not_at_stop():
    # long, stop 98, next open gaps to 97: must not book a profit at the stop price
    b = bars_from([[100, 100, 100, 100], [97, 97.5, 96, 97]])
    t = simulate(b, [Signal(bar=0, side=1, stop=98, target=105)], ZERO_COST)
    assert t.iloc[0]["exit_reason"] == "stop" and t.iloc[0]["gross_ret"] <= 0


def test_to_utc_handles_mixed_ms_and_us_rows():
    import pandas as pd
    from quant_lab.data.binance import _to_utc
    ms = 1735689600000  # 2025-01-01T00:00Z in ms
    out = _to_utc(pd.Series([ms - 3_600_000, ms * 1000]))
    assert list(out) == [pd.Timestamp("2024-12-31T23:00Z"), pd.Timestamp("2025-01-01T00:00Z")]


def test_write_result_parallel_writers_keep_both_keys(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    import json
    import quant_lab.results as res
    monkeypatch.setattr(res, "RESULTS_DIR", tmp_path)
    with ThreadPoolExecutor(8) as ex:
        list(ex.map(lambda k: res.write_result("epx", f"k{k}", {"v": k}, params={}, sources=[]), range(16)))
    assert len(json.loads((tmp_path / "epx.json").read_text())["experiments"]) == 16

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

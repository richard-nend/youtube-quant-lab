import importlib.util

import pandas as pd

from quant_lab import LAB_ROOT

_spec = importlib.util.spec_from_file_location("meta_label", LAB_ROOT / "episodes/ep08/meta_label.py")
ml = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ml)


def ts(s):
    return pd.Timestamp(s, tz="UTC")


def test_label_is_one_for_losing_or_flat_trade():
    assert ml.label_of(-0.001) == 1
    assert ml.label_of(0.0) == 1
    assert ml.label_of(0.002) == 0


def test_purge_excludes_trades_exiting_after_fold_start():
    rows = pd.DataFrame({
        "key": ["before", "straddle", "after"],
        "signal_time": [ts("2024-10-01"), ts("2024-10-30"), ts("2024-11-02")],
        "exit_time": [ts("2024-10-02"), ts("2024-11-03"), ts("2024-11-04")],
    })
    train, test = ml.split(rows, ts("2024-11-01"), ts("2025-05-01"))
    assert list(train.key) == ["before"]
    assert list(test.key) == ["after"]


def test_last_fold_end_is_exclusive():
    rows = pd.DataFrame({
        "key": ["in", "out"],
        "signal_time": [ts("2026-10-31T23:00"), ts("2026-11-01T00:00")],
        "exit_time": [ts("2026-11-01T02:00"), ts("2026-11-01T03:00")],
    })
    _, test = ml.split(rows, ml.FOLD_STARTS[-1], ml.FOLD_END_LAST)
    assert list(test.key) == ["in"]


def _state(i, side="long", strategy="fair value gap limit entry", z=0.0):
    return {"strategy": strategy, "side": side, "ret_24h_pct": float(i % 7), "ret_7d_pct": float(i % 5),
            "atr_pct": 1.0 + (i % 3), "delta_z": z, "from_20d_high_pct": -float(i % 4), "from_20d_low_pct": float(i % 9)}


def _frame(states, labels):
    return pd.DataFrame({"state": states, "label": labels})


def test_logistic_test_row_does_not_change_other_predictions():
    train = _frame([_state(i, z=float(i % 2)) for i in range(60)], [i % 2 for i in range(60)])
    test = _frame([_state(i) for i in range(5)], [0] * 5)
    p1 = ml.logistic_scores(train, test)
    moved = test.copy()
    moved.at[0, "state"] = _state(0, z=1e6)
    p2 = ml.logistic_scores(train, moved)
    assert (abs(p1[1:] - p2[1:]) < 1e-12).all()


def test_logistic_uses_strategy_and_side():
    # label is fully explained by side; numeric fields carry no signal
    states = [_state(i, side="long" if i % 2 else "short") for i in range(80)]
    train = _frame(states, [i % 2 for i in range(80)])
    test = _frame([_state(1, side="long"), _state(1, side="short")], [1, 0])
    p = ml.logistic_scores(train, test)
    assert p[0] > 0.7 and p[1] < 0.3


def test_fold_ends_follow_starts():
    ends = ml.fold_ends()
    assert ends[:-1] == ml.FOLD_STARTS[1:]
    assert ends[-1] == ml.FOLD_END_LAST


def _sym(b=True, c=True, e=True, kept=200):
    return {"b": b, "c": c, "e": e, "kept_trades": kept}


def test_verdict_pass_needs_both_symbols():
    assert ml.verdict({"BTCUSDT": _sym(), "ETHUSDT": _sym()})["verdict"] == "PASS"


def test_verdict_weak_is_btc_only():
    assert ml.verdict({"BTCUSDT": _sym(), "ETHUSDT": _sym(c=False)})["verdict"] == "WEAK"


def test_verdict_fail_when_drawdown_better_but_expectancy_not():
    assert ml.verdict({"BTCUSDT": _sym(c=False), "ETHUSDT": _sym()})["verdict"] == "FAIL"


def test_verdict_fail_when_logistic_wins():
    assert ml.verdict({"BTCUSDT": _sym(e=False), "ETHUSDT": _sym()})["verdict"] == "FAIL"


def test_verdict_insufficient_below_100_kept_trades():
    v = ml.verdict({"BTCUSDT": _sym(kept=99), "ETHUSDT": _sym()})
    assert v["verdict"] == "FAIL" and v["per_symbol"]["BTCUSDT"] == "insufficient"


def test_loss_size_treats_gains_as_zero():
    assert ml.loss_size(-0.2) == 0.2
    assert ml.loss_size(0.05) == 0.0

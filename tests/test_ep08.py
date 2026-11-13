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


def test_fold_ends_follow_starts():
    ends = ml.fold_ends()
    assert ends[:-1] == ml.FOLD_STARTS[1:]
    assert ends[-1] == ml.FOLD_END_LAST

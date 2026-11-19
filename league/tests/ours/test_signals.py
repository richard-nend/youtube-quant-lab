import numpy as np
import pandas as pd
import pytest

from league.ours import signals as sg

pytestmark = pytest.mark.skipif(not (sg.MAIN_ROOT / ".cache" / "binance").exists(),
                                reason=f"no Binance archive cache under QL_MAIN_ROOT={sg.MAIN_ROOT}")


def _same(a, b):
    if a is None or b is None:
        return a is None and b is None
    return (np.isnan(a) and np.isnan(b)) or abs(a - b) <= 1e-9


def test_live_facts_match_build_trades():
    """20 past signals: facts from bars cut at the signal bar == EP08b build_trades x (built on the full history)."""
    ref = pd.concat([sg.mb.build_trades(sym, sg.asm) for sym in sg.SYMBOLS], ignore_index=True)
    pick = ref.iloc[np.random.default_rng(0).choice(len(ref), 20, replace=False)]
    full = {(sym, tf): sg.binance.klines(sym, tf, sg.asm.START, sg.asm.END, market="um")
            for sym in sg.SYMBOLS for tf in ("1h", "4h")}
    for r in pick.itertuples():
        b = full[(r.sym, sg.asm.CANDIDATES[r.candidate][0])]
        b = b[b.index <= r.signal_time]  # only bars closed by the signal
        sigs, trades = sg.run_candidate(b, r.candidate)
        got = sg.live_facts(r.sym, r.candidate, r.signal_time, b, trades, sigs)
        assert got.keys() == r.x.keys()
        bad = {k: (got[k], r.x[k]) for k in got if not _same(got[k], r.x[k])}
        assert not bad, (r.sym, r.candidate, r.signal_time, bad)


def test_chosen_candidate_needs_30_trades():
    m = pd.Timestamp("2025-01-01", tz="UTC")
    t = lambda n, r: pd.DataFrame({"signal_time": [m - pd.Timedelta(days=10)] * n, "net_ret": [r] * n})  # noqa: E731
    assert sg.chosen_candidate({"A": t(30, 0.001), "B": t(29, 0.01)}, m) == "A"
    assert sg.chosen_candidate({"A": t(29, 0.001)}, m) is None

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


def test_new_signals_hourly_steps_match_runner_rule(monkeypatch):
    """Step `now` hourly across the Feb->Mar 2026 boundary on archive bars (no REST); BTC switches FVG-4h -> FVG-1h.
    Each signal must come out exactly once, in the hour its bar closes, and equal what the live runner's own
    chosen_candidate admits from the bars known at that hour (its month choice uses provisional open trades)."""
    t0, t1 = pd.Timestamp("2026-02-28 08:00", tz="UTC"), pd.Timestamp("2026-03-01 20:00", tz="UTC")
    full = {(sym, tf): sg.binance.klines(sym, tf, sg.asm.START, "2026-03-05", market="um")
            for sym in sg.SYMBOLS for tf in ("1h", "4h")}
    step = {"1h": pd.Timedelta(hours=1), "4h": pd.Timedelta(hours=4)}
    cut = lambda sym, tf, now: full[(sym, tf)][full[(sym, tf)].index + step[tf] <= now]  # noqa: E731  closed bars
    monkeypatch.setattr(sg.runner, "bars", cut)

    got, want = [], set()
    for now in pd.date_range(t0 + pd.Timedelta(hours=1), t1, freq="h"):
        for s in sg.new_signals(now):
            close = s.signal_time + step[sg.asm.CANDIDATES[s.candidate][0]]
            assert now - pd.Timedelta(hours=1) < close <= now, (now, s.key)
            got.append(s.key)
        for sym in sg.SYMBOLS:
            runs = {c: sg.run_candidate(cut(sym, tf, now), c) for c, (tf, *_) in sg.asm.CANDIDATES.items()}
            trades = {c: t for c, (_, t) in runs.items()}
            for cand, (tf, *_) in sg.asm.CANDIDATES.items():
                last = cut(sym, tf, now).index[-1]  # the bar that closed this hour, if any
                month = pd.Timestamp(last.year, last.month, 1, tz="UTC")  # the signal's month
                if last + step[tf] > now - pd.Timedelta(hours=1) and sg.runner.chosen_candidate(trades, month) == cand:
                    want |= {f"{sym}|{cand}|{last.isoformat()}" for x in runs[cand][0]
                             if cut(sym, tf, now).index[x.bar] == last}
    assert len(got) == len(set(got)), "a signal was emitted twice"
    assert set(got) == want
    assert {sg.asm.CANDIDATES[k.split("|")[1]][0] for k in want} == {"1h", "4h"}  # both bar sizes exercised
    assert {k.split("|")[1] for k in want if k.startswith("BTCUSDT")} == {"FVG-4h", "FVG-1h"}  # month switch seen


def test_chosen_candidate_needs_30_trades():
    m = pd.Timestamp("2025-01-01", tz="UTC")
    t = lambda n, r: pd.DataFrame({"signal_time": [m - pd.Timedelta(days=10)] * n, "net_ret": [r] * n})  # noqa: E731
    assert sg.chosen_candidate({"A": t(30, 0.001), "B": t(29, 0.01)}, m) == "A"
    assert sg.chosen_candidate({"A": t(29, 0.001)}, m) is None

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


def test_new_signals_hourly_steps_match_weekly_rule(monkeypatch, tmp_path):
    """Step `now` hourly across the Feb->Mar 2026 month boundary and the Mon 2026-03-02 week boundary on archive
    bars (no REST). Each signal comes out exactly once, in the hour its bar closes, and only for the candidate frozen
    for its week (runner rule on trades closed by Monday 00:00 UTC); within a week each symbol uses one candidate."""
    t0, t1 = pd.Timestamp("2026-02-28 08:00", tz="UTC"), pd.Timestamp("2026-03-02 12:00", tz="UTC")
    full = {(sym, tf): sg.binance.klines(sym, tf, sg.asm.START, "2026-03-05", market="um")
            for sym in sg.SYMBOLS for tf in ("1h", "4h")}
    step = {"1h": pd.Timedelta(hours=1), "4h": pd.Timedelta(hours=4)}
    cut = lambda sym, tf, now: full[(sym, tf)][full[(sym, tf)].index + step[tf] <= now]  # noqa: E731  closed bars
    monkeypatch.setattr(sg.runner, "bars", cut)

    got, want, week = [], set(), tmp_path / "week.json"
    for now in pd.date_range(t0 + pd.Timedelta(hours=1), t1, freq="h"):
        for s in sg.new_signals(now, week):
            close = s.signal_time + step[sg.asm.CANDIDATES[s.candidate][0]]
            assert now - pd.Timedelta(hours=1) < close <= now, (now, s.key)
            got.append(s.key)
        for sym in sg.SYMBOLS:
            runs = {c: sg.run_candidate(cut(sym, tf, now), c) for c, (tf, *_) in sg.asm.CANDIDATES.items()}
            trades = {c: t for c, (_, t) in runs.items()}
            for cand, (tf, *_) in sg.asm.CANDIDATES.items():
                last = cut(sym, tf, now).index[-1]  # the bar that closed this hour, if any
                wk = sg.week_start(last)
                closed = {c: t[t.exit_time < wk] for c, t in trades.items()}
                if last + step[tf] > now - pd.Timedelta(hours=1) and sg.runner.chosen_candidate(closed, wk) == cand:
                    want |= {f"{sym}|{cand}|{last.isoformat()}" for x in runs[cand][0]
                             if cut(sym, tf, now).index[x.bar] == last}
    assert len(got) == len(set(got)), "a signal was emitted twice"
    assert set(got) == want and want
    per_week = {}
    for k in got:
        sym, cand, ts = k.split("|")
        per_week.setdefault((sym, sg.week_start(pd.Timestamp(ts))), set()).add(cand)
    assert all(len(c) == 1 for c in per_week.values()), per_week
    assert {w for _, w in per_week} == {pd.Timestamp("2026-02-23", tz="UTC"), pd.Timestamp("2026-03-02", tz="UTC")}


def test_chosen_candidate_needs_30_trades():
    m = pd.Timestamp("2025-01-01", tz="UTC")
    t = lambda n, r: pd.DataFrame({"signal_time": [m - pd.Timedelta(days=10)] * n, "net_ret": [r] * n})  # noqa: E731
    assert sg.chosen_candidate({"A": t(30, 0.001), "B": t(29, 0.01)}, m) == "A"
    assert sg.chosen_candidate({"A": t(29, 0.001)}, m) is None


def test_week_choice_frozen_within_week_changes_at_boundary(tmp_path):
    """리그 전용: 후보는 월요일 00:00 UTC에 끝난 거래만으로 정하고 그 주 내내 고정, 재시작해도 같다."""
    mon = pd.Timestamp("2026-03-02", tz="UTC")                       # 월요일
    assert sg.week_start(mon + pd.Timedelta(days=6, hours=23)) == mon and sg.week_start(mon - pd.Timedelta(hours=1)) == mon - pd.Timedelta(days=7)

    def t(n, r, exit_at, sig_at=mon - pd.Timedelta(days=20)):
        return pd.DataFrame({"signal_time": [sig_at] * n, "exit_time": [exit_at] * n, "net_ret": [r] * n})
    early = mon - pd.Timedelta(days=1)
    trades = {"A": t(30, 0.001, early), "B": t(30, 0.0005, early)}
    path = tmp_path / "week.json"
    assert sg.week_choice(trades, "BTCUSDT", mon + pd.Timedelta(hours=1), path) == "A"
    # 주 중간에 B의 큰 수익 거래가 끝나도(또는 열린 거래가 잠정 수익을 보여도) 이번 주는 A
    trades["B"] = pd.concat([trades["B"], t(30, 0.05, mon + pd.Timedelta(days=2), mon + pd.Timedelta(days=1))])
    trades["B"] = pd.concat([trades["B"], t(30, 0.05, mon + pd.Timedelta(days=9), mon - pd.Timedelta(days=3))])   # 월요일 전 신호지만 아직 열림
    for h in (2, 50, 100, 167):
        assert sg.week_choice(trades, "BTCUSDT", mon + pd.Timedelta(hours=h), path) == "A"
    assert sg.week_choice(trades, "BTCUSDT", mon + pd.Timedelta(hours=5), None) == "A"   # 끝난 거래만: 열린 거래는 안 센다
    assert sg.week_choice(trades, "BTCUSDT", mon + pd.Timedelta(days=7), path) == "B"     # 다음 월요일에 바뀐다
    restarted = {"A": t(30, 0.9, early), "B": t(1, 0.0, early)}                         # 재시작 뒤 다시 계산하지 않는다
    assert sg.week_choice(restarted, "BTCUSDT", mon + pd.Timedelta(hours=3), path) == "A"
    assert sg.week_choice(restarted, "BTCUSDT", mon + pd.Timedelta(days=8), path) == "B"

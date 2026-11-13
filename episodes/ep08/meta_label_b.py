"""EP08b (H-EP08b): Laya used as its docs advise, given every fact known before entry.

Differences from H-EP08 (meta_label.py): English checkpoint, full fine-tune (on Kaggle GPU), training pool = all four
candidates' trades (L0), a richer English sentence state, the question phrased as the label, and a rank rule
(veto the top 20%) instead of p >= 0.5. Test trades are still EP07's L1 trades.

    uv run --directory quant_lab python episodes/ep08/meta_label_b.py export
"""

from __future__ import annotations

import importlib.util
import json
import sys

import numpy as np
import pandas as pd

from quant_lab import LAB_ROOT

SYMS = ("BTCUSDT", "ETHUSDT")
SEED = 20261113
VETO_SHARE = 0.20
Q_LOSS = "Will this trade lose money after costs?"
RUNS = LAB_ROOT / "runs" / "ep08b"
KAGGLE_DIR = RUNS / "kaggle_data"
WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
NAMES = {"FVG-1h": "fair value gap pullback, 1-hour bars", "FVG-4h": "fair value gap pullback, 4-hour bars",
         "FLOW-1h": "breakout with aggressive buying, 1-hour bars", "FLOW-4h": "breakout with aggressive buying, 4-hour bars"}


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, LAB_ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ml = _load("meta_label", "episodes/ep08/meta_label.py")  # FOLD_STARTS, FOLD_END_LAST, fold_ends, split, label_of


def exante_cost(entry: str, costs) -> float:
    """Round-trip cost known before entry: entry by order type, exit assumed taker (the engine's cost depends on
    how the trade ends, so it must not be used as a feature)."""
    return costs.side_cost("maker" if entry == "limit" else "taker") + costs.side_cost("taker")


def recent_record(trades: pd.DataFrame, signal_time: pd.Timestamp, n: int = 20) -> tuple[float | None, float | None, int]:
    """Win rate and mean net return of the last n trades of the same candidate that closed before the signal."""
    done = trades[trades.exit_time < signal_time].tail(n)
    if done.empty:
        return None, None, 0
    return float((done.net_ret > 0).mean()), float(done.net_ret.mean()), len(done)


def _context(b: pd.DataFrame, asm) -> pd.DataFrame:
    c = b.close
    per_day = 24 if (b.index[1] - b.index[0]) == pd.Timedelta(hours=1) else 6
    base = asm.state_features(b, 720 if per_day == 24 else 180)
    atr_pct = base["atr_pct"]
    base["ret_30d_pct"] = 100 * (c / c.shift(30 * per_day) - 1)
    base["atr_pctile_90d"] = 100 * atr_pct.rolling(90 * per_day).rank(pct=True)
    base["from_ma200_pct"] = 100 * (c / c.rolling(200).mean() - 1)
    return base


def build_trades(sym: str, asm) -> pd.DataFrame:
    """All L0 trades for one symbol with ex-ante features (no ticker, date or price level)."""
    from quant_lab.data import binance
    from quant_lab.engine.backtest import simulate

    bars = {tf: binance.klines(sym, tf, asm.START, asm.END, market="um") for tf in ("1h", "4h")}
    out = []
    for name, (tf, zwin, fn, sim) in asm.CANDIDATES.items():
        b = bars[tf]
        sigs = fn(b) if fn else asm.flow_signals(b, zwin)
        t = simulate(b, sigs, asm.COSTS, **sim)
        by_time = {}
        for s in sigs:
            by_time.setdefault(b.index[s.bar], s)
        ctx = _context(b, asm)
        hours = 1 if tf == "1h" else 4
        max_hold = sim.get("max_hold", 96)
        for _, r in t.iterrows():
            s = by_time[r.signal_time]
            k = b.index.get_loc(r.signal_time)
            ref = s.limit_price if s.entry == "limit" else float(b.close.iloc[k])
            stop_pct = 100 * abs(ref - s.stop) / ref
            target_pct = 100 * abs(s.target - ref) / ref
            cost_pct = 100 * exante_cost(s.entry, asm.COSTS)
            m = r.signal_time.to_period("M").to_timestamp().tz_localize("UTC")
            w = asm.month_slice(t, m - pd.DateOffset(months=12), m)
            wr, mean_net, n_rec = recent_record(t, r.signal_time)
            f = ctx.loc[r.signal_time]
            out.append({
                "sym": sym, "candidate": name, "signal_time": r.signal_time, "entry_time": r.entry_time,
                "exit_time": r.exit_time, "gross_ret": r.gross_ret, "net_ret": r.net_ret, "side": int(r.side),
                "label": ml.label_of(r.net_ret),
                "x": {"long": float(r.side > 0), "limit_entry": float(s.entry == "limit"),
                      "stop_pct": stop_pct, "target_pct": target_pct, "reward_risk": target_pct / stop_pct,
                      "cost_pct": cost_pct, "cost_to_stop": cost_pct / stop_pct, "max_hold_h": max_hold * hours,
                      "trail12_sum_pct": 100 * float(w.net_ret.sum()), "trail12_n": float(len(w)),
                      "recent_win": wr, "recent_mean_pct": None if mean_net is None else 100 * mean_net,
                      "hour": float(r.signal_time.hour), "weekday": float(r.signal_time.weekday()),
                      **{k2: (None if pd.isna(v) else float(v)) for k2, v in f.items()},
                      **{f"cand_{c}": float(c == name) for c in asm.CANDIDATES}},
            })
    return pd.DataFrame(out)


def _num(v, nd=2, pct=False):
    return "unknown" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{nd}f}{'%' if pct else ''}"


def state_text(r: dict, candidate: str) -> str:
    x = r
    side = "long (buy)" if x["long"] else "short (sell)"
    entry = "a limit order (maker fee)" if x["limit_entry"] else "a market order (taker fee)"
    rec = ("no earlier closed trades of this strategy" if x["recent_win"] is None else
           f"of its last trades, {_num(100 * x['recent_win'], 0, True)} won and the average net return was "
           f"{_num(x['recent_mean_pct'], 3, True)}")
    return (
        f"Strategy: {NAMES[candidate]}. Direction: {side}, entered with {entry}. "
        f"The stop is {_num(x['stop_pct'], 2, True)} away and the target {_num(x['target_pct'], 2, True)} away, "
        f"a reward-to-risk ratio of {_num(x['reward_risk'])}. Round-trip costs are about {_num(x['cost_pct'], 3, True)}, "
        f"which is {_num(100 * x['cost_to_stop'], 0, True)} of the stop distance. The trade is closed after at most "
        f"{int(x['max_hold_h'])} hours. "
        f"Market: {_num(x['ret_24h_pct'], 2, True)} over 24 hours, {_num(x['ret_7d_pct'], 2, True)} over 7 days, "
        f"{_num(x['ret_30d_pct'], 2, True)} over 30 days. Volatility (ATR) is {_num(x['atr_pct'], 2, True)} of price, "
        f"at the {_num(x['atr_pctile_90d'], 0)}th percentile of the last 90 days. Price is "
        f"{_num(x['from_ma200_pct'], 2, True)} from its 200-bar average, {_num(x['from_20d_high_pct'], 2, True)} from the "
        f"20-day high and {_num(x['from_20d_low_pct'], 2, True)} from the 20-day low. Aggressive-buying z-score on the "
        f"signal bar: {_num(x['delta_z'])}. "
        f"This strategy's record: {rec}; over the previous 12 months it made {_num(x['trail12_sum_pct'], 1, True)} "
        f"net across {int(x['trail12_n'])} trades. "
        f"Time: {int(x['hour']):02d}:00 UTC on a {WEEKDAYS[int(x['weekday'])]}."
    )


def load_all() -> pd.DataFrame:
    l1mod = _load("l1_split", "episodes/ep07/l1_split.py")
    asm = l1mod.asm
    parts = []
    for sym in SYMS:
        t = build_trades(sym, asm)
        l1 = l1mod.l1_trades(sym)
        l1_keys = set(l1.signal_time)
        # L1 trades are L0 trades of the month's chosen candidate; mark them by (candidate, signal_time) via EP07's cache
        cache = ml.load_rows()
        chosen = set(cache[cache.sym == sym].key)
        t["key"] = sym + "|" + t.candidate + "|" + t.signal_time.map(lambda x: x.isoformat())
        t["is_l1"] = t.key.isin(chosen)
        assert t.is_l1.sum() == len(l1) == len(chosen), (sym, int(t.is_l1.sum()), len(l1), len(chosen))
        assert set(t[t.is_l1].signal_time) == l1_keys
        parts.append(t)
    rows = pd.concat(parts, ignore_index=True)
    rows["state"] = [state_text(x, c) for x, c in zip(rows.x, rows.candidate)]
    return rows


def export() -> None:
    rows = load_all()
    KAGGLE_DIR.mkdir(parents=True, exist_ok=True)
    meta = {}
    for start, end in zip(ml.FOLD_STARTS, ml.fold_ends()):
        train = rows[rows.exit_time < start]                                   # all candidates, purged
        test = rows[rows.is_l1 & (rows.signal_time >= start) & (rows.signal_time < end)]
        assert not set(train.key) & set(test.key)
        tag = f"{start:%Y-%m}"
        with (KAGGLE_DIR / f"train_{tag}.jsonl").open("w") as fh:
            for s, y in zip(train.state, train.label):
                fh.write(json.dumps({"state": s, "questions": {"a": {"type": "noul", "instructions": Q_LOSS}},
                                     "expected": {"a": int(y)}}) + "\n")
        with (KAGGLE_DIR / f"test_{tag}.jsonl").open("w") as fh:
            for k, s in zip(test.key, test.state):
                fh.write(json.dumps({"key": k, "state": s}) + "\n")
        meta[tag] = {"train_rows": len(train), "train_label1": float(train.label.mean()), "test_rows": len(test)}
        print(tag, meta[tag], flush=True)
    (KAGGLE_DIR / "folds.json").write_text(json.dumps({"question": Q_LOSS, "folds": meta}, indent=1))
    rows.assign(signal_time=rows.signal_time.astype(str), entry_time=rows.entry_time.astype(str),
                                    exit_time=rows.exit_time.astype(str)).to_json(RUNS / "rows.jsonl", orient="records", lines=True)
    print(rows.state.iloc[0])


if __name__ == "__main__":
    {"export": export}[sys.argv[1] if len(sys.argv) > 1 else "export"]()

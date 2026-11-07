"""Live paper test (PREREGISTRATION.md H-EP07-LIVE + Amendment 3). Paper only: no orders, no keys.

Every hour: rebuild Binance bars (archive + public REST, closed bars only), replay the exact EP07B
procedure with the backtest engine, record new signals (with Laya scores) before their outcome is
known, and recompute the four tracks from the trades the engine produces.

    uv run --with vendor/laya/laya-0.4.22-py3-none-any.whl python live/runner.py
"""

from __future__ import annotations

import importlib.util
import json
import os
import time
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from quant_lab import LAB_ROOT
from quant_lab.data import binance
from quant_lab.engine.backtest import simulate

LIVE = LAB_ROOT / "live"
START = pd.Timestamp(json.loads((LIVE / "start.json").read_text())["start_utc"])
SYMBOLS = ("BTCUSDT", "ETHUSDT")
LATE_AFTER = timedelta(minutes=70)
EQUITY0 = 10_000.0
LAYA = {"multi": "model", "en": "model-en"}  # track 1 / track 3 brakes
REST = "https://fapi.binance.com/fapi/v1/klines"

_spec = importlib.util.spec_from_file_location("assembly", LAB_ROOT / "episodes/ep07/assembly_walkforward.py")
asm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(asm)


def rest_klines(symbol, interval, start_ms, now_ms):
    rows = []
    while True:
        q = f"{REST}?symbol={symbol}&interval={interval}&startTime={start_ms}&limit=1500"
        batch = json.load(urllib.request.urlopen(q, timeout=30))
        rows += [r for r in batch if r[6] < now_ms]  # closed bars only
        if len(batch) < 1500:
            return rows
        start_ms = batch[-1][0] + 1


def bars(symbol, interval, now):
    """Archive up to 2 days ago (daily files lag), then REST for the rest; same columns as binance.klines."""
    arch_end = (now - timedelta(days=2)).date()
    b = binance.klines(symbol, interval, asm.START, arch_end, market="um")
    rows = rest_klines(symbol, interval, int(b.index[-1].timestamp() * 1000) + 1, int(now.timestamp() * 1000))
    if rows:
        r = pd.DataFrame(rows, columns=binance.KLINE_COLS)
        r.index = pd.to_datetime(r.open_time, unit="ms", utc=True)
        r = r[["open", "high", "low", "close", "volume", "quote_volume", "taker_buy_volume", "taker_buy_quote_volume"]].astype(float)
        b = pd.concat([b[r.columns.intersection(b.columns)], r])
    b = b[~b.index.duplicated()].sort_index()
    step = b.index[1] - b.index[0]
    # the archive itself has holes (e.g. 2026-10-28 is absent from the Oct-2026 monthly file): fill them from REST
    d = b.index.to_series().diff()
    for t0 in d[d > step].index:
        prev = b.index[b.index.get_loc(t0) - 1]
        gap = rest_klines(symbol, interval, int((prev + step).timestamp() * 1000), int(t0.timestamp() * 1000))
        if gap:
            g = pd.DataFrame(gap, columns=binance.KLINE_COLS)
            g.index = pd.to_datetime(g.open_time, unit="ms", utc=True)
            b = pd.concat([b, g[b.columns.intersection(g.columns)].astype(float)])
    b = b[~b.index.duplicated()].sort_index()
    assert (b.index.to_series().diff().dropna() == step).all(), f"gap in {symbol} {interval} bars"
    return b


def chosen_candidate(trades, month):
    lo = month - pd.DateOffset(months=12)
    score = {n: (float(asm.month_slice(t, lo, month).net_ret.sum()) if len(asm.month_slice(t, lo, month)) >= 30 else -np.inf)
             for n, t in trades.items()}
    best = max(score, key=score.get)
    return None if score[best] == -np.inf else best


def load_jsonl(p):
    return [json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []


def laya_scores(states):
    """{key: {multi: p, en: p}} for new states only; models loaded only when needed."""
    if not states:
        return {}
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    import laya
    out = {k: {} for k in states}
    for tag, ckpt in LAYA.items():
        agent = laya.load(str(LAB_ROOT / "vendor" / "laya" / ckpt), device="cpu")
        for k, s in states.items():
            out[k][tag] = agent.predict(s, {"a": {"type": "noul", "instructions": asm.Q1}})["answers"]["a"]["noul"]
    return out


def track(trades, veto=None, risk=None):
    t = trades if veto is None else trades[~veto]
    if t.empty:
        return {"trades": 0, "equity": EQUITY0}
    frac = np.ones(len(t)) if risk is None else np.minimum(risk / t.risk.to_numpy(), 3.0)
    eq = EQUITY0 * np.cumprod(1 + frac * t.net_ret.to_numpy())
    return {"trades": int(len(t)), "wins": int((t.net_ret > 0).sum()), "mean_net": float(t.net_ret.mean()),
            "equity": round(float(eq[-1]), 2), "max_dd": float((eq / np.maximum.accumulate(np.r_[EQUITY0, eq])[1:] - 1).min())}


def main():
    now = datetime.now(timezone.utc)
    sig_path = LIVE / "signals.jsonl"
    seen = {r["key"]: r for r in load_jsonl(sig_path)}
    new_rows, all_trades, status = [], [], {"run_utc": now.isoformat(timespec="seconds"), "start_utc": START.isoformat()}

    for sym in SYMBOLS:
        bb = {tf: bars(sym, tf, now) for tf in ("1h", "4h")}
        trades, sigs, feats = {}, {}, {}
        for name, (tf, zwin, fn, sim) in asm.CANDIDATES.items():
            b = bb[tf]
            s = fn(b) if fn else asm.flow_signals(b, zwin)
            t = simulate(b, s, asm.COSTS, **sim)
            t["candidate"], t["symbol"] = name, sym
            # an exit stamped on the last bar by "time" before max_hold elapsed is the data end, not an exit
            last = b.index[-1]
            open_ = (t.exit_reason == "time") & (t.exit_time == last) & \
                    (t.entry_time.map(b.index.get_loc) + sim["max_hold"] > len(b) - 1)
            t["open"] = open_.to_numpy()
            trades[name], sigs[name], feats[name] = t, (b, s), asm.state_features(b, zwin)
            status[f"{sym}_{tf}_last_bar"] = last.isoformat()

        months = pd.date_range(START.to_period("M").to_timestamp().tz_localize("UTC"), now, freq="MS")
        if len(months) == 0 or months[0] > START:
            months = months.insert(0, START.to_period("M").to_timestamp().tz_localize("UTC"))
        status[f"{sym}_chosen"] = {}
        for m in months:
            cand = chosen_candidate(trades, m)
            status[f"{sym}_chosen"][m.strftime("%Y-%m")] = cand
            if cand is None:
                continue
            m_end = m + pd.DateOffset(months=1)
            b, s = sigs[cand]
            for sg in s:
                ts = b.index[sg.bar]
                if not (max(m, START) <= ts < m_end):
                    continue
                key = f"{sym}|{cand}|{ts.isoformat()}"
                if key in seen:
                    continue
                bar_close = ts + (b.index[1] - b.index[0])
                f = feats[cand].loc[ts]
                state = {"strategy": "fair value gap limit entry" if cand.startswith("FVG") else "breakout with aggressive buying",
                         "side": "long" if sg.side > 0 else "short",
                         **{k: (None if pd.isna(v) else round(float(v), 2)) for k, v in f.items()}}
                new_rows.append({"key": key, "symbol": sym, "candidate": cand, "signal_bar": ts.isoformat(),
                                 "recorded_utc": now.isoformat(timespec="seconds"), "late": bool(now - bar_close > LATE_AFTER),
                                 "side": int(sg.side), "entry": sg.entry, "limit_price": sg.limit_price,
                                 "stop": round(float(sg.stop), 2), "target": round(float(sg.target), 2), "state": state})
            mt = asm.month_slice(trades[cand], max(m, START), m_end)
            all_trades.append(mt)

    scores = laya_scores({r["key"]: r["state"] for r in new_rows})
    with sig_path.open("a") as fh:
        for r in new_rows:
            r["laya_noul"] = {k: round(v, 4) for k, v in scores[r["key"]].items()}
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            seen[r["key"]] = r

    t = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    if not t.empty:
        t["key"] = t.symbol + "|" + t.candidate + "|" + t.signal_time.map(lambda x: x.isoformat())
        t["late"] = t.key.map(lambda k: seen[k]["late"])
        t["veto_multi"] = t.key.map(lambda k: seen[k]["laya_noul"]["multi"] >= 0.5)
        t["veto_en"] = t.key.map(lambda k: seen[k]["laya_noul"]["en"] >= 0.5)
        t = t.sort_values("exit_time").reset_index(drop=True)
        closed = t[~t.open]
        tracks = {"1_registered_multi_2pct": track(closed, closed.veto_multi.to_numpy(), 0.02),
                  "2_walkforward_only": track(closed),
                  "3_laya_en": track(closed, closed.veto_en.to_numpy()),
                  "4_risk_0.5pct_exploratory": track(closed, risk=0.005)}
        cols = ["key", "symbol", "candidate", "side", "signal_time", "entry_time", "exit_time", "entry_type", "entry_px",
                "exit_px", "exit_reason", "gross_ret", "cost", "net_ret", "risk", "open", "late", "veto_multi", "veto_en"]
        t[cols].to_csv(LIVE / "trades.csv", index=False)
    else:
        tracks = {}
    status["new_signals"] = len(new_rows)
    status["tracks"] = tracks
    (LIVE / "status.json").write_text(json.dumps(status, indent=1, ensure_ascii=False, default=str) + "\n")
    print(json.dumps({"new_signals": len(new_rows), "tracks": tracks}, default=str))


if __name__ == "__main__":
    main()

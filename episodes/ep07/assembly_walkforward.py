"""EP07B — H-EP07B in PREREGISTRATION.md (commit 73cc6ff): assemble the EP01–EP06 survivors.

L0 each candidate alone · L1 monthly walk-forward selection · L2 + Laya brake (vs random veto) · L3 + 2% risk sizing.
Run:  uv run --with vendor/laya/laya-0.4.22-py3-none-any.whl python episodes/ep07/assembly_walkforward.py
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from quant_lab import LAB_ROOT
from quant_lab.data import binance
from quant_lab.engine.backtest import simulate
from quant_lab.engine.costs import CostModel
from quant_lab.protocol import audit, trade_stats, verdict
from quant_lab.results import write_result

HERE = Path(__file__).resolve().parent
EPS = HERE.parent
# Amendment 3 re-test: LAYA_CKPT=model-en | model-typed (default = the registered multilingual checkpoint)
LAYA_CKPT = os.environ.get("LAYA_CKPT", "model")
LAYA_DIR = LAB_ROOT / "vendor" / "laya" / LAYA_CKPT
CACHE = LAB_ROOT / "runs" / ("ep07_laya_answers.jsonl" if LAYA_CKPT == "model" else f"ep07_laya_answers_{LAYA_CKPT}.jsonl")
RESULT_KEY = "assembly_walkforward" + ("" if LAYA_CKPT == "model" else f"_{LAYA_CKPT.replace('model-', 'laya_')}")
COSTS = CostModel()
START, END = "2022-11-01", "2026-10-31"
MONTHS = pd.date_range("2023-11-01", "2026-10-01", freq="MS", tz="UTC")
SEED, N_RAND, N_BOOT = 20261107, 20, 10_000
Q1 = "Should this trade entry be blocked because conditions are dangerous for it?"
Q2 = "Is this a bad moment to open this position?"


def load(name, rel):
    spec = importlib.util.spec_from_file_location(name, EPS / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


fvg = load("cost_aware_smc", "ep03/cost_aware_smc.py")
flow = load("flow_confirmed_breakout", "ep06/flow_confirmed_breakout.py")

FVG_SIM = dict(fvg.SIM)
CANDIDATES = {  # name -> (interval, z window in bars, signal fn, sim kwargs)
    "FVG-1h": ("1h", 720, lambda b: fvg.fvg_signals(b, False), FVG_SIM),
    "FVG-4h": ("4h", 180, lambda b: fvg.fvg_signals(b, False), FVG_SIM),
    "FLOW-1h": ("1h", 720, None, {"max_hold": 48}),
    "FLOW-4h": ("4h", 180, None, {"max_hold": 12}),
}


def flow_signals(b, zwin):
    old = flow.ZWIN
    flow.ZWIN = zwin  # same rule, z window = 30 days in bars of this interval
    try:
        return flow.breakout_signals(b, "T1")
    finally:
        flow.ZWIN = old


def state_features(b, zwin):
    """Past-only facts per bar (no ticker, date or price level)."""
    c = b.close
    bars_per_day = 24 if (b.index[1] - b.index[0]) == pd.Timedelta(hours=1) else 6
    a = flow.atr(b)
    delta = 2 * b.taker_buy_volume - b.volume
    past = delta.shift(1).rolling(zwin)
    return pd.DataFrame({
        "ret_24h_pct": 100 * (c / c.shift(bars_per_day) - 1),
        "ret_7d_pct": 100 * (c / c.shift(7 * bars_per_day) - 1),
        "atr_pct": 100 * a / c,
        "delta_z": (delta - past.mean()) / past.std(),
        "from_20d_high_pct": 100 * (c / b.high.rolling(20 * bars_per_day).max() - 1),
        "from_20d_low_pct": 100 * (c / b.low.rolling(20 * bars_per_day).min() - 1),
    }, index=b.index)


def month_slice(t, m0, m1):
    return t[(t.signal_time >= m0) & (t.signal_time < m1)]


def curve_stats(t, frac=None):
    """Equity at fixed notional (frac=None) or with per-trade fractions; worst month, max drawdown."""
    if t.empty:
        return {"final": 1.0, "max_dd": 0.0, "worst_month": 0.0}
    f = np.ones(len(t)) if frac is None else frac
    eq = np.cumprod(1 + f * t.net_ret.to_numpy())
    peak = np.maximum.accumulate(np.concatenate([[1.0], eq]))[1:]
    mret = pd.Series(f * t.net_ret.to_numpy(), index=pd.DatetimeIndex(t.exit_time)).groupby(
        lambda x: x.strftime("%Y-%m")).apply(lambda r: float(np.prod(1 + r) - 1))
    return {"final": float(eq[-1]), "max_dd": float((eq / peak - 1).min()), "worst_month": float(mret.min()),
            "curve": np.round(eq, 4).tolist()}


def card(t):
    """Report-card checks that apply to a concatenated walk-forward trade list.
    Delay and holdout are not applicable here (the walk-forward is itself out-of-sample)."""
    if t.empty:
        return {"flags": ["no trades"]}
    st = trade_stats(t)
    flags = []
    ratio = float((t.cost / t.risk).median())
    if ratio > 0.10:
        flags.append(f"cost/risk {ratio:.0%}")
    exp2 = float((t.net_ret - t.cost).mean())
    lo, hi = st["ci95"]
    if st["trades"] < 100 or lo <= 0 <= hi:
        flags.append("sample/CI")
    net = t.net_ret.to_numpy()
    if net.sum() > 0 and np.sort(net)[-5:].sum() / net.sum() > 0.5:
        flags.append("concentration")
    return {"cost_to_risk_median": ratio, "expectancy_2x_costs": exp2, "flags": flags}


def laya_answers(rows):
    """rows: list of (key, state). Cached so a re-run does not re-ask."""
    done = {}
    if CACHE.exists():
        for line in CACHE.read_text().splitlines():
            r = json.loads(line)
            done[r["key"]] = r
    todo = [(k, s) for k, s in rows if k not in done]
    if todo:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        import laya
        agent = laya.load(str(LAYA_DIR), device="cpu")
        with CACHE.open("a") as fh:
            for n, (k, s) in enumerate(todo):
                r = {"key": k, "state": s}
                for tag, q in (("q1", Q1), ("q2", Q2)):
                    r[tag] = agent.predict(s, {"a": {"type": "noul", "instructions": q}})["answers"]["a"]["noul"]
                fh.write(json.dumps(r) + "\n")
                done[k] = r
                if n % 100 == 0:
                    print(f"laya {n}/{len(todo)}", flush=True)
    return done


def main():
    out = {"months": [m.strftime("%Y-%m") for m in MONTHS]}
    rng = np.random.default_rng(SEED)
    for sym in ("BTCUSDT", "ETHUSDT"):
        bars = {tf: binance.klines(sym, tf, START, END, market="um") for tf in ("1h", "4h")}
        trades, feats, l0 = {}, {}, {}
        for name, (tf, zwin, fn, sim) in CANDIDATES.items():
            b = bars[tf]
            sigs = fn(b) if fn else flow_signals(b, zwin)
            # shortcut: each candidate is simulated continuously; L1 takes its trades whose signal falls in the chosen month
            t = simulate(b, sigs, COSTS, **sim)
            t["candidate"] = name
            trades[name], feats[name] = t, state_features(b, zwin)
            oos = t[t.signal_time >= MONTHS[0]]
            sig_oos = [s for s in sigs if b.index[s.bar] >= MONTHS[0]]
            a = audit(f"{sym} {name}", b, sig_oos, COSTS, sim)  # delay / cost / 2x / sample / concentration on the 36 months
            l0[name] = {"stats": trade_stats(oos), "curve": curve_stats(oos), "audit_flags": a["flags"]}
            print(sym, name, l0[name]["stats"].get("trades"), l0[name]["stats"].get("expectancy"), a["flags"], flush=True)

        # L1 — monthly selection by trailing-12-month net sum
        chosen, l1 = [], []
        for m in MONTHS:
            lo, hi = m - pd.DateOffset(months=12), m + pd.DateOffset(months=1)
            score = {}
            for name, t in trades.items():
                w = month_slice(t, lo, m)
                score[name] = float(w.net_ret.sum()) if len(w) >= 30 else -np.inf
            best = max(score, key=score.get)
            if score[best] == -np.inf:
                chosen.append({"month": m.strftime("%Y-%m"), "candidate": None})
                continue
            mt = month_slice(trades[best], m, hi)
            l1.append(mt)
            chosen.append({"month": m.strftime("%Y-%m"), "candidate": best, "trailing_sum": score[best], "trades": len(mt)})
        t1 = pd.concat(l1, ignore_index=True).sort_values("entry_time").reset_index(drop=True)

        # L2 — Laya brake on every L1 entry
        rows = []
        for i, r in t1.iterrows():
            f = feats[r.candidate].loc[r.signal_time]
            st = {"strategy": "fair value gap limit entry" if r.candidate.startswith("FVG") else "breakout with aggressive buying",
                  "side": "long" if r.side > 0 else "short",
                  **{k: (None if pd.isna(v) else round(float(v), 2)) for k, v in f.items()}}
            rows.append((f"{sym}|{r.candidate}|{r.signal_time.isoformat()}", st))
        ans = laya_answers(rows)
        p1 = np.array([ans[k]["q1"] for k, _ in rows])
        p2 = np.array([ans[k]["q2"] for k, _ in rows])
        veto = p1 >= 0.5
        t2 = t1[~veto].reset_index(drop=True)
        rand = []
        for _ in range(N_RAND):
            keep = rng.random(len(t1)) >= veto.mean()
            tr = t1[keep]
            rand.append({"expectancy": float(tr.net_ret.mean()) if len(tr) else None, **{k: v for k, v in curve_stats(tr).items() if k != "curve"}})
        t2b = t1[~(p2 >= 0.5)]

        # L3 — 2% risk per trade, leverage cap 3x, on the L2 trades
        frac = np.minimum(0.02 / t2.risk.to_numpy(), 3.0)
        c3 = curve_stats(t2, frac)
        months_l3 = [g for _, g in t2.assign(_f=frac).groupby(t2.exit_time.dt.strftime("%Y-%m"))]
        brng = np.random.default_rng(SEED)
        dd = []
        for _ in range(N_BOOT):
            seq = pd.concat([months_l3[j] for j in brng.integers(0, len(months_l3), len(months_l3))])
            eq = np.cumprod(1 + seq._f.to_numpy() * seq.net_ret.to_numpy())
            dd.append(float((eq / np.maximum.accumulate(np.concatenate([[1.0], eq]))[1:] - 1).min()))
        dd = np.array(dd)

        out[sym] = {
            "L0": l0,
            "L1": {"stats": trade_stats(t1), "curve": curve_stats(t1), "card": card(t1), "chosen": chosen},
            "L2": {"veto_rate": float(veto.mean()), "stats": trade_stats(t2), "curve": curve_stats(t2), "card": card(t2),
                   "random_veto": {"runs": rand,
                                   "median_expectancy": float(np.median([r["expectancy"] for r in rand])),
                                   "median_max_dd": float(np.median([r["max_dd"] for r in rand])),
                                   "median_worst_month": float(np.median([r["worst_month"] for r in rand]))},
                   "q2_robustness": {"veto_rate": float((p2 >= 0.5).mean()), "stats": trade_stats(t2b),
                                     "curve": {k: v for k, v in curve_stats(t2b).items() if k != "curve"}},
                   "vetoed_mean_net": float(t1[veto].net_ret.mean()) if veto.any() else None,
                   "kept_mean_net": float(t2.net_ret.mean()) if len(t2) else None},
            "L3": {"curve": c3, "median_frac": float(np.median(frac)), "p_dd_50": float((dd <= -0.5).mean()),
                   "dd_ci": [float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5))]},
        }
        print(sym, "L1", trade_stats(t1)["expectancy"], "L2 veto", round(float(veto.mean()), 3), trade_stats(t2).get("expectancy"),
              "rand med", out[sym]["L2"]["random_veto"]["median_expectancy"], "L3", c3["final"], c3["max_dd"], flush=True)

    out["verdict_L1"] = verdict({s: out[s]["L1"]["stats"] for s in ("BTCUSDT", "ETHUSDT")}, primary="BTCUSDT")
    out["verdict_L2"] = verdict({s: out[s]["L2"]["stats"] for s in ("BTCUSDT", "ETHUSDT")}, primary="BTCUSDT")
    write_result("ep07", RESULT_KEY, out,
                 params={"candidates": {k: v[0] for k, v in CANDIDATES.items()}, "months": "2023-11..2026-10",
                         "laya": {"model": "convaiinnovations/laya-multilingual@1720e3e3", "model-en": "convaiinnovations/laya@7b928d82", "model-typed": "convaiinnovations/laya-typed-decisions@e929ae5c"}[LAYA_CKPT] + ", laya 0.4.22, cpu", "q1": Q1, "q2": Q2,
                         "veto_threshold": 0.5, "n_random": N_RAND, "risk_per_trade": 0.02, "leverage_cap": 3,
                         "costs": COSTS.as_dict(), "preregistration": "PREREGISTRATION.md H-EP07B (commit 73cc6ff)"},
                 seed=SEED)


if __name__ == "__main__":
    main()

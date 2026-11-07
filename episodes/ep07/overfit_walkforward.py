"""EP07 — H-EP07 in PREREGISTRATION.md (commit 8d0d773): grid-search illusion, permutation test, walk-forward.

1h SMA cross, market entry next open, stop k*ATR14, target r*k*ATR14, max hold 48.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from quant_lab.data import binance
from quant_lab.engine.backtest import Signal, simulate
from quant_lab.engine.costs import ZERO_COST, CostModel
from quant_lab.protocol import trade_stats, verdict
from quant_lab.results import write_result

COSTS = CostModel()
HOLD, MIN_TRADES, N_PERM, SEED = 48, 30, 200, 20261107
GRID = [(f, s, k, r) for f, s, k, r in itertools.product(
    (5, 10, 15, 20, 30, 40, 50), (20, 30, 50, 80, 120, 160, 200), (1, 1.5, 2, 3), (1, 2, 3)) if f < s]
IS, OOS = ("2024-11-01", "2025-10-31"), ("2025-11-01", "2026-10-31")
WARM = 200  # longest SMA; signals start after it


def atr(b, n=14):
    tr = pd.concat([b.high - b.low, (b.high - b.close.shift()).abs(), (b.low - b.close.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean().to_numpy()


def signals(b, combo, lo=0, hi=None) -> list[Signal]:
    """Cross signals with signal bar in [lo, hi)."""
    f, s, k, r = combo
    c = b.close
    x = np.sign((c.rolling(f).mean() - c.rolling(s).mean()).to_numpy())
    a, cc = atr(b), c.to_numpy()
    hi = len(b) if hi is None else hi
    out = []
    for i in range(max(lo, WARM, 1), hi):
        if x[i] != 0 and x[i] != x[i - 1] and not np.isnan(x[i - 1]):
            side = int(x[i])
            out.append(Signal(i, side, cc[i] - side * k * a[i], cc[i] + side * r * k * a[i]))
    return out


def score(t):
    return float(t["net_ret"].sum()) if len(t) >= MIN_TRADES else -np.inf


def run(b, combo, costs, lo=0, hi=None):
    return simulate(b, signals(b, combo, lo, hi), costs, max_hold=HOLD)


def gap_through_stop(b, t):
    """Trades whose entry open was already beyond the stop (engine counts them oddly; reported, not removed)."""
    if t.empty:
        return 0
    return int(((t.exit_reason == "stop") & (t.gross_ret > 0)).sum())


def best(b, costs, lo=0, hi=None):
    scores = {g: score(run(b, g, costs, lo, hi)) for g in GRID}
    g = max(scores, key=scores.get)
    return g, scores


def shuffled(b, rng):
    """Shuffle whole bars (OHLC relative to previous close) and rebuild a price path."""
    pc = b.close.shift().bfill().to_numpy()
    rel = np.column_stack([b.open / pc, b.high / pc, b.low / pc, b.close / pc])
    rel = rel[rng.permutation(len(rel))]
    close = b.close.iloc[0] * np.cumprod(rel[:, 3])
    prev = np.concatenate([[b.close.iloc[0]], close[:-1]])
    return pd.DataFrame({"open": rel[:, 0] * prev, "high": rel[:, 1] * prev, "low": rel[:, 2] * prev,
                         "close": close}, index=b.index)


def walk_forward(b, first_month: str, last_month: str):
    months = pd.date_range(first_month, last_month, freq="MS", tz="UTC")
    trades, chosen = [], []
    for m in months:
        lo_t, hi_t = m - pd.DateOffset(months=12), m + pd.DateOffset(months=1)
        lo, mid, hi = (int(b.index.searchsorted(x)) for x in (lo_t, m, hi_t))
        g, scores = best(b, COSTS, lo, mid)
        if scores[g] == -np.inf:
            chosen.append({"month": m.strftime("%Y-%m"), "combo": None})
            continue
        is_t = run(b, g, COSTS, lo, mid)
        t = run(b, g, COSTS, mid, hi)
        trades.append(t)
        chosen.append({"month": m.strftime("%Y-%m"), "combo": list(g), "is_trades": len(is_t),
                       "is_mean": float(is_t.net_ret.mean()), "oos_trades": len(t),
                       "oos_mean": float(t.net_ret.mean()) if len(t) else None})
    t = pd.concat(trades, ignore_index=True) if trades else pd.DataFrame()
    st = trade_stats(t)
    picked = [c for c in chosen if c["combo"]]
    st["is_mean_of_chosen"] = float(np.average([c["is_mean"] for c in picked], weights=[c["is_trades"] for c in picked]))
    st["equity_curve"] = np.round(np.cumprod(1 + t.net_ret.to_numpy()), 4).tolist() if len(t) else []
    st["gap_through_stop"] = gap_through_stop(b, t)
    return st, chosen


def main():
    btc = binance.klines("BTCUSDT", "1h", "2022-11-01", OOS[1], market="um")
    eth = binance.klines("ETHUSDT", "1h", "2022-11-01", OOS[1], market="um")
    b_is = btc.loc[IS[0]:IS[1]]
    b_oos = btc.loc[btc.index >= pd.Timestamp(OOS[0], tz="UTC") - pd.Timedelta(hours=WARM + 1)]
    oos_lo = int(b_oos.index.searchsorted(pd.Timestamp(OOS[0], tz="UTC")))
    out = {"grid_size": len(GRID)}

    # E1 — the illusion
    g0, s0 = best(b_is, ZERO_COST)
    _, s_cost = best(b_is, COSTS)
    t_is0, t_is, t_oos = run(b_is, g0, ZERO_COST), run(b_is, g0, COSTS), run(b_oos, g0, COSTS, oos_lo)
    ok0 = [v for v in s0.values() if v > -np.inf]
    okc = [v for v in s_cost.values() if v > -np.inf]
    out["E1"] = {"best_combo_zero_cost": {"fast": g0[0], "slow": g0[1], "stop_atr": g0[2], "rr": g0[3]},
                 "is_zero_cost": trade_stats(t_is0), "is_real_cost": trade_stats(t_is), "oos_real_cost": trade_stats(t_oos),
                 "combos_scored": len(ok0),
                 "share_positive_zero_cost": float(np.mean([v > 0 for v in ok0])),
                 "share_positive_real_cost": float(np.mean([v > 0 for v in okc])),
                 "curves": {"is_zero_cost": np.round(np.cumprod(1 + t_is0.net_ret.to_numpy()), 4).tolist(),
                            "is_real_cost": np.round(np.cumprod(1 + t_is.net_ret.to_numpy()), 4).tolist(),
                            "oos_real_cost": np.round(np.cumprod(1 + t_oos.net_ret.to_numpy()), 4).tolist()},
                 "gap_through_stop": gap_through_stop(b_is, t_is) + gap_through_stop(b_oos, t_oos)}
    print("E1", out["E1"]["best_combo_zero_cost"], {k: out["E1"][k]["expectancy"] for k in ("is_zero_cost", "is_real_cost", "oos_real_cost")}, flush=True)

    # E2 — permutation test on the IS year
    rng = np.random.default_rng(SEED)
    real_best = s0[g0]
    noise = []
    for i in range(N_PERM):
        _, sp = best(shuffled(b_is, rng), ZERO_COST)
        noise.append(max(sp.values()))
        if i % 20 == 0:
            print(f"perm {i}: noise best {noise[-1]:+.3f} vs real {real_best:+.3f}", flush=True)
    noise = np.array(noise)
    out["E2"] = {"real_best_sum_net_zero_cost": real_best, "noise_best": np.round(noise, 5).tolist(),
                 "p_value": float((noise >= real_best).mean()), "noise_median": float(np.median(noise))}
    print("E2 p =", out["E2"]["p_value"], flush=True)

    # E3 — walk-forward
    out["E3"] = {}
    for name, b in (("BTC", btc), ("ETH", eth)):
        st, chosen = walk_forward(b, "2023-11-01", "2026-10-01")
        out["E3"][name] = {"stats": st, "months": chosen}
        print("E3", name, {k: st[k] for k in ("trades", "expectancy", "ci95", "is_mean_of_chosen")}, flush=True)
    vs = {k: v["stats"] for k, v in out["E3"].items()}
    out["E3"]["verdict"] = verdict(vs, primary="BTC")

    write_result("ep07", "overfit_walkforward", out,
                 params={"grid": "fast×slow×k×r, fast<slow", "grid_size": len(GRID), "hold": HOLD,
                         "min_trades": MIN_TRADES, "n_perm": N_PERM, "is": IS, "oos": OOS,
                         "costs": COSTS.as_dict(), "preregistration": "PREREGISTRATION.md H-EP07 (commit 8d0d773)"},
                 seed=SEED)


if __name__ == "__main__":
    main()

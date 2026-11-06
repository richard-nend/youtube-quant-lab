"""Backtest audit ("report card") — the checks pre-registered in PREREGISTRATION.md (H-EP01).

Every strategy in the series goes through the same six checks:
  1 delay      shift every signal one bar later; >50% drop in net return -> flag
  2 cost/risk  median round-trip cost > 10% of 1R -> flag
  3 2x costs   net expectancy with doubled costs
  4 sample     < 100 trades or bootstrap 95% CI of mean net return includes 0 -> flag
  5 holdout    sign of net expectancy flips on the holdout period -> flag
  6 concentr.  5 best trades > 50% of total net profit -> flag
The thresholds are rules of thumb, not statistical laws.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Callable

import numpy as np
import pandas as pd

from quant_lab.engine.backtest import Signal, simulate
from quant_lab.engine.costs import CostModel

BOOT_SEED = 20261106
BOOT_N = 10_000


def bootstrap_ci(x, n: int = BOOT_N, seed: int = BOOT_SEED) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), size=(n, len(x)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def trade_stats(t: pd.DataFrame) -> dict:
    if t.empty:
        return {"trades": 0}
    r = t["net_ret"].to_numpy()
    lo, hi = bootstrap_ci(r)
    return {"trades": int(len(r)), "win_rate": float((r > 0).mean()), "expectancy": float(r.mean()),
            "expectancy_before_costs": float(t["gross_ret"].mean()),
            "expectancy_r": float(t["r_multiple"].mean()) if "r_multiple" in t else None,
            "ci95": [lo, hi], "sum_net": float(r.sum())}


def verdict(per_set: dict[str, dict], primary: str = "BTC-H", min_trades: int = 100) -> str:
    """PASS / WEAK / FAIL as defined in PREREGISTRATION.md."""
    ok = {k: v.get("trades", 0) >= min_trades and v.get("expectancy", -1) > 0 for k, v in per_set.items()}
    if all(ok.values()):
        return "PASS"
    if ok.get(primary):
        return "WEAK"
    return "FAIL"


def delay(signals: list[Signal], bars: int = 1) -> list[Signal]:
    return [replace(s, bar=s.bar + bars) for s in signals]


def audit(name: str, bars: pd.DataFrame, signals: list[Signal], costs: CostModel, sim_kwargs: dict,
          holdout: tuple[pd.DataFrame, list[Signal]] | None = None) -> dict:
    """Run the six checks. signals must be computed causally from `bars` (and holdout from its bars)."""
    t = simulate(bars, signals, costs, **sim_kwargs)
    base = trade_stats(t)
    out = {"name": name, "base": base, "flags": []}
    if t.empty:
        out["flags"].append("no trades")
        return out

    # 1 delay
    td = simulate(bars, delay(signals), costs, **sim_kwargs)
    d_sum = float(td["net_ret"].sum()) if not td.empty else 0.0
    out["delay_1bar"] = {"sum_net": d_sum, "expectancy": float(td["net_ret"].mean()) if not td.empty else None}
    if base["sum_net"] > 0 and d_sum < 0.5 * base["sum_net"]:
        out["flags"].append("delay: edge collapses when acting one bar later")

    # 2 cost / risk  (risk = |entry - stop| / entry)
    ratio = float((t["cost"] / t["risk"]).median())
    out["cost_to_risk_median"] = ratio
    if ratio > 0.10:
        out["flags"].append(f"cost/risk: costs are {ratio:.0%} of 1R")

    # 3 2x costs
    c2 = CostModel(costs.taker_fee * 2, costs.maker_fee * 2, costs.slippage_bps * 2)
    t2 = simulate(bars, signals, c2, **sim_kwargs)
    out["expectancy_2x_costs"] = float(t2["net_ret"].mean()) if not t2.empty else None

    # 4 sample
    lo, hi = base["ci95"]
    if base["trades"] < 100 or (lo <= 0 <= hi):
        out["flags"].append(f"sample: {base['trades']} trades, 95% CI [{lo:+.3%}, {hi:+.3%}] includes 0"
                            if lo <= 0 <= hi else f"sample: only {base['trades']} trades")

    # 5 holdout
    if holdout is not None:
        th = simulate(holdout[0], holdout[1], costs, **sim_kwargs)
        hs = trade_stats(th)
        out["holdout"] = hs
        if hs.get("trades", 0) and np.sign(hs["expectancy"]) != np.sign(base["expectancy"]):
            out["flags"].append("holdout: sign of expectancy flips")

    # 6 concentration
    net = t["net_ret"].to_numpy()
    if net.sum() > 0:
        top5 = float(np.sort(net)[-5:].sum() / net.sum())
        out["top5_share"] = top5
        if top5 > 0.5:
            out["flags"].append(f"concentration: 5 best trades = {top5:.0%} of profit")
    return out


def run_sets(make: Callable[[pd.DataFrame], list[Signal]], sets: dict[str, pd.DataFrame], costs: CostModel,
             sim_kwargs: dict) -> dict[str, dict]:
    """Apply one signal generator to several (instrument, period) bar sets."""
    return {k: trade_stats(simulate(b, make(b), costs, **sim_kwargs)) for k, b in sets.items()}

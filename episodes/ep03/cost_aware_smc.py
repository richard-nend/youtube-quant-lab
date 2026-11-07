"""EP03 answer experiment — H-EP03 in PREREGISTRATION.md (cost-to-risk rule).

Round-1 B_causal_fvg_only signals, unchanged, on 15m / 1h / 4h, with and without the rule
"skip unless stop distance >= 10 x round_trip(maker entry, taker stop)".
Sets: BTC-D (2025-11 -> 2026-10, seen), BTC-H (2024-11 -> 2025-10, holdout), ETH-H (2024-11 -> 2026-10).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from quant_lab.data import binance
from quant_lab.engine.backtest import simulate
from quant_lab.engine.costs import CostModel
from quant_lab.protocol import run_sets, verdict
from quant_lab.results import write_result

_spec = importlib.util.spec_from_file_location("smc_backtest", Path(__file__).with_name("smc_backtest.py"))
smc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(smc)

COSTS = CostModel()
RT = COSTS.round_trip("maker", "taker")
COST_MULT = 10.0
SIM = {"fill": "through", "limit_ttl": 20, "max_hold": 96}
SETS = {"BTC-D": ("BTCUSDT", "2025-11-01", "2026-10-31"),
        "BTC-H": ("BTCUSDT", "2024-11-01", "2025-10-31"),
        "ETH-H": ("ETHUSDT", "2024-11-01", "2026-10-31")}


def fvg_signals(b, rule: bool):
    l, h = b.low.to_numpy(), b.high.to_numpy()
    sigs = [s for s in (smc.fvg_signal(z, l, h) for z in smc.causal_fvgs(b)) if s]
    if rule:
        sigs = [s for s in sigs if abs(s.limit_price - s.stop) / s.limit_price >= COST_MULT * RT]
    return sigs


def main():
    out = {"round_trip_maker_taker": RT, "min_stop_pct": COST_MULT * RT}
    for tf in ("15m", "1h", "4h"):
        sets = {k: binance.klines(sym, tf, s, e, market="um") for k, (sym, s, e) in SETS.items()}
        for rule in (False, True):
            name = f"{tf}_{'rule' if rule else 'no_rule'}"
            res = run_sets(lambda b: fvg_signals(b, rule), sets, COSTS, SIM)
            res["verdict"] = verdict(res)
            out[name] = res
            print(f"{name:12} {res['verdict']:5} " + "  ".join(
                f"{k}: n={v['trades']:>5} win={v.get('win_rate', 0):.1%} exp={v.get('expectancy', 0):+.3%} "
                f"pre={v.get('expectancy_before_costs', 0):+.3%} R={v.get('expectancy_r') or 0:+.2f}"
                for k, v in res.items() if k != "verdict"), flush=True)
    # ---- chart data for the video (no effect on the numbers above) ----------------------
    # per-trade cost / stop-distance for 15m no-rule on BTC-D (same trades as report_card "EP03 FVG mechanical (15m)")
    sym, s0, e0 = SETS["BTC-D"]
    b = binance.klines(sym, "15m", s0, e0, market="um")
    t = simulate(b, fvg_signals(b, False), COSTS, **SIM)
    assert len(t) == out["15m_no_rule"]["BTC-D"]["trades"]
    ratio = (t["cost"] / t["risk"]).to_numpy()
    edges = np.round(np.arange(0, 2.0001, 0.05), 2)
    cnt, _ = np.histogram(np.clip(ratio, 0, 2.0), bins=edges)
    out["charts"] = {"cost_to_risk_15m_BTC-D": {
        "trades": int(len(t)), "bin_edges": edges.tolist(), "counts": cnt.astype(int).tolist(),
        "note": "last bin includes ratios >= 1.95 (clipped at 2.0)",
        "median": round(float(np.median(ratio)), 4),
        "share_risk_ge_rule": round(float((t["risk"] >= COST_MULT * RT).mean()), 4)}}

    write_result("ep03", "cost_aware_smc", out,
                 params={"sets": SETS, "cost_mult": COST_MULT, "rr": smc.RR, "sim": SIM, "costs": COSTS.as_dict(),
                         "preregistration": "PREREGISTRATION.md H-EP03 (commit 197476e)"})


if __name__ == "__main__":
    main()

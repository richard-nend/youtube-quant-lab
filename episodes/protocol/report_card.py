"""H-EP01 — the backtest report card: the same six checks on every strategy in the series.

See quant_lab/protocol.py and PREREGISTRATION.md (H-EP01).
Discovery = BTC 2025-11 -> 2026-10 (what the episodes showed); holdout = BTC 2024-11 -> 2025-10.
EP01 strategies are always-in-market position flips (no stop), so they get the equivalent checks
computed directly: delay = act one bar later; cost/risk = cost vs median |gross trade return|.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from quant_lab.data import binance
from quant_lab.engine.costs import CostModel
from quant_lab.protocol import audit, bootstrap_ci
from quant_lab.results import write_result

EP = Path(__file__).resolve().parents[1]
D = ("2025-11-01", "2026-10-31")
H = ("2024-11-01", "2025-10-31")
COSTS = CostModel()


def load(rel: str):
    spec = importlib.util.spec_from_file_location(Path(rel).stem, EP / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# ------------------------------------------------------------------ EP01 (position-flip strategies)
def ep01_trades(m, b: pd.DataFrame, which: str, lag: int) -> pd.DataFrame:
    """lag 0 = what the chart shows; lag 1 = act at the next bar's real open."""
    n = len(b)
    if which == "ha":
        ha = m.heikin_ashi(b)
        sig = np.where(ha.close > ha.open, 1, -1)
        if lag == 0:
            return m.trades_from_positions(sig, ha.close.to_numpy())
        return m.trades_from_positions(np.concatenate([[0], sig[:-1]]), b.open.to_numpy())
    h, l = b.high.to_numpy(), b.low.to_numpy()
    K = m.K
    piv = np.zeros(n, dtype=int)
    for p in range(K, n - K):
        if l[p] == l[p - K:p + K + 1].min():
            piv[p] = 1
        elif h[p] == h[p - K:p + K + 1].max():
            piv[p] = -1
    pos, px, cur = np.zeros(n, int), np.zeros(n), 0
    if lag == 0:
        for i in range(n):
            if piv[i]:
                cur = piv[i]
            pos[i] = cur
            px[i] = l[i] if piv[i] == 1 else (h[i] if piv[i] == -1 else b.close.iloc[i])
        return m.trades_from_positions(pos, px)
    for i in range(n):
        p = i - K - 1
        if p >= 0 and piv[p]:
            cur = piv[p]
        pos[i] = cur
    return m.trades_from_positions(pos, b.open.to_numpy())


def ep01_audit(m, name, which, bd, bh):
    rt = COSTS.round_trip("taker", "taker")
    shown = ep01_trades(m, bd, which, 0)
    live = ep01_trades(m, bd, which, 1)
    hold = ep01_trades(m, bh, which, 0)
    net_shown = shown.gross_ret                       # the chart shows no fees
    net_live = live.gross_ret - rt
    lo, hi = bootstrap_ci(net_live)
    flags = []
    shown_sum, live_sum = float(net_shown.sum()), float(net_live.sum())
    if shown_sum > 0 and live_sum < 0.5 * shown_sum:
        flags.append("delay: edge collapses when acting one bar later")
    ratio = float(rt / live.gross_ret.abs().median())
    if ratio > 0.10:
        flags.append(f"cost/risk: costs are {ratio:.0%} of a typical trade move")
    if len(live) < 100 or lo <= 0 <= hi:
        flags.append(f"sample: 95% CI [{lo:+.3%}, {hi:+.3%}] includes 0" if lo <= 0 <= hi else "sample: < 100 trades")
    if np.sign(hold.gross_ret.mean()) != np.sign(net_shown.mean()):
        flags.append("holdout: sign of expectancy flips")
    return {"name": name, "base": {"trades": int(len(shown)), "expectancy": float(net_shown.mean()),
                                   "win_rate": float((net_shown > 0).mean())},
            "delay_1bar": {"trades": int(len(live)), "expectancy": float(net_live.mean()),
                           "expectancy_before_costs": float(live.gross_ret.mean()),
                           "win_rate": float((net_live > 0).mean()), "ci95": [lo, hi]},
            "cost_to_risk_median": ratio,
            "expectancy_2x_costs": float((live.gross_ret - 2 * rt).mean()),
            "holdout": {"trades": int(len(hold)), "expectancy_as_shown": float(hold.gross_ret.mean())},
            "flags": flags}


def main():
    cards = []
    ep01 = load("ep01/fake_backtest_tricks.py")
    b1d, b1h = (binance.klines("BTCUSDT", "1h", *p, market="um") for p in (D, H))
    cards.append(ep01_audit(ep01, "EP01 Heikin-Ashi colour flip", "ha", b1d, b1h))
    cards.append(ep01_audit(ep01, "EP01 repainting pivot arrows", "pivot", b1d, b1h))

    smc = load("ep03/smc_backtest.py")
    cas = load("ep03/cost_aware_smc.py")
    b15d, b15h = (binance.klines("BTCUSDT", "15m", *p, market="um") for p in (D, H))
    sd, sh = smc.filtered_signals(b15d), smc.filtered_signals(b15h)
    smc_sim = {"fill": "through", "limit_ttl": 20, "max_hold": 96}
    cards.append(audit("EP03 FVG mechanical (15m)", b15d, sd["B_causal_fvg_only"], COSTS, smc_sim,
                       holdout=(b15h, sh["B_causal_fvg_only"])))
    cards.append(audit("EP03 FVG + 5 filters (15m)", b15d, sd["C5_+cvd_absorption"], COSTS, smc_sim,
                       holdout=(b15h, sh["C5_+cvd_absorption"])))
    cards.append(audit("EP03 FVG + cost/risk rule (15m)", b15d, cas.fvg_signals(b15d, True), COSTS, smc_sim,
                       holdout=(b15h, cas.fvg_signals(b15h, True))))
    b1hd = binance.klines("BTCUSDT", "1h", *D, market="um")
    cards.append(audit("EP03 FVG mechanical (1h)", b1hd, cas.fvg_signals(b1hd, False), COSTS, smc_sim,
                       holdout=(b1h, cas.fvg_signals(b1h, False))))

    ab = load("ep06/absorption_bot_backtest.py")
    b5d, b5h = (binance.klines("BTCUSDT", "5m", *p, market="um") for p in (D, H))
    _, v1d, _, v4d, _ = ab.signals(b5d)
    _, v1h, _, v4h, _ = ab.signals(b5h)
    cards.append(audit("EP06 absorption fade, market orders (5m)", b5d, v1d, COSTS, {"max_hold": ab.HOLD},
                       holdout=(b5h, v1h)))
    cards.append(audit("EP06 failure entry + 5x cost rule (5m)", b5d, v4d, COSTS, {"max_hold": ab.HOLD * 2},
                       holdout=(b5h, v4h)))

    fl = load("ep06/flow_confirmed_breakout.py")
    def period(p):
        b = binance.klines("BTCUSDT", "1h", (pd.Timestamp(p[0]) - pd.Timedelta(days=31)).date(), p[1], market="um")
        start = pd.Timestamp(p[0], tz=b.index.tz)
        return b, lambda mode: [s for s in fl.breakout_signals(b, mode) if b.index[s.bar] >= start]
    (bd_, sd_), (bh_, sh_) = period(D), period(H)
    for mode, label in (("T0", "EP06 1h breakout, no order flow"), ("T1", "EP06 1h breakout + aggression filter")):
        cards.append(audit(label, bd_, sd_(mode), COSTS, fl.SIM, holdout=(bh_, sh_(mode))))

    for c in cards:
        b = c["base"]
        hold = c.get("holdout", {})
        print(f"{c['name']:42} n={b.get('trades', 0):>5} exp={b.get('expectancy', 0):+.3%} "
              f"hold={hold.get('expectancy', hold.get('expectancy_as_shown', 0)):+.3%}  flags={len(c['flags'])}")
        for f in c["flags"]:
            print("      -", f)
    write_result("protocol", "report_card", {"cards": cards},
                 params={"discovery": D, "holdout": H, "costs": COSTS.as_dict(),
                         "preregistration": "PREREGISTRATION.md H-EP01 (commit 197476e)"})


if __name__ == "__main__":
    main()

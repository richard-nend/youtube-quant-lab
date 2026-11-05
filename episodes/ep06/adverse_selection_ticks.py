"""EP06 — "my limit order only fills when I'm wrong" (adverse selection), measured on ticks.

Pre-registered design (BTCUSDT perp aggTrades, first day of each month Jan–Jun 2026):
  every 60 s, place a passive BUY limit at (last trade price - OFFSET) and a passive SELL limit at
  (last trade price + OFFSET), each valid for TTL seconds.
  fill rule "touch"  : a trade prints at or beyond the limit
  fill rule "through": a trade prints strictly beyond the limit by >= 1 tick (0.1 USD) —
                       a stand-in for "the queue ahead of you was fully consumed"
  outcome            : signed return from the limit price to the trade price H seconds after the
                       ORDER WAS PLACED (same horizon for filled and unfilled, so no look-ahead in
                       the comparison). For unfilled orders this is the profit you missed.
Adverse selection shows up as filled orders having a worse forward return than unfilled ones.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_lab.data import binance
from quant_lab.results import write_result

DAYS = [f"2026-{m:02d}-01" for m in range(1, 7)]
OFFSET, TTL, H, TICK = 1.0, 60, 300, 0.1


def one_day(day: str) -> pd.DataFrame:
    t = binance.agg_trades("BTCUSDT", day)
    ts = t.index.as_unit("ms").asi8               # epoch milliseconds
    px = t["price"].to_numpy()
    rows = []
    start, end = ts[0] + 60_000, ts[-1] - (H + 5) * 1000
    for place in range(int(start), int(end), 60_000):
        i0 = np.searchsorted(ts, place, side="right") - 1
        last = px[i0]
        j_ttl = np.searchsorted(ts, place + TTL * 1000, side="right")
        j_h = np.searchsorted(ts, place + H * 1000, side="right") - 1
        window = px[i0 + 1: j_ttl]
        p_h = px[j_h]
        for side in (1, -1):                      # +1 buy below, -1 sell above
            lim = last - side * OFFSET
            touch = bool(window.size and ((window <= lim).any() if side > 0 else (window >= lim).any()))
            thru = bool(window.size and ((window <= lim - TICK).any() if side > 0 else (window >= lim + TICK).any()))
            rows.append({"day": day, "side": side, "touch": touch, "through": thru,
                         "fwd_ret_bp": side * (p_h - lim) / lim * 1e4})
    return pd.DataFrame(rows)


def main():
    df = pd.concat([one_day(d) for d in DAYS], ignore_index=True)
    out = {"orders": int(len(df)), "offset_usd": OFFSET, "ttl_s": TTL, "horizon_s": H}
    for rule in ("touch", "through"):
        f, nf = df[df[rule]], df[~df[rule]]
        out[rule] = {"fill_rate": float(df[rule].mean()),
                     "filled_mean_fwd_bp": float(f.fwd_ret_bp.mean()),
                     "unfilled_mean_fwd_bp": float(nf.fwd_ret_bp.mean()),
                     "filled_share_profitable": float((f.fwd_ret_bp > 0).mean()),
                     "unfilled_share_would_have_profited": float((nf.fwd_ret_bp > 0).mean())}
        x = out[rule]
        print(f"{rule:8} fill {x['fill_rate']:.1%} | filled fwd {x['filled_mean_fwd_bp']:+.2f}bp "
              f"({x['filled_share_profitable']:.0%} profitable) | unfilled missed {x['unfilled_mean_fwd_bp']:+.2f}bp "
              f"({x['unfilled_share_would_have_profited']:.0%} would have profited)")
    write_result("ep06", "adverse_selection_ticks", out,
                 params={"days": DAYS, "offset_usd": OFFSET, "ttl_s": TTL, "horizon_s": H, "tick": TICK},
                 note="fees excluded: this isolates selection, maker fee would subtract a further 2bp")


if __name__ == "__main__":
    main()

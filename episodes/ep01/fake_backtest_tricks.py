"""EP01 — two backtest tricks behind "95% win rate" indicators, measured on real data.

1) Heikin-Ashi fills: the same HA colour-flip strategy, filled at (a) synthetic HA prices
   (what a strategy run on an HA chart reports unless real-OHLC fills are forced) vs
   (b) real next-bar open prices, with and without fees.
2) Repainting / look-ahead: "buy at the pivot low, sell at the pivot high" where a pivot needs
   K bars on each side. (a) arrow drawn on the pivot bar itself (what a repainting indicator
   shows on history) vs (b) the same signal acted on when it is actually confirmed (K bars later).

Data: BTCUSDT USDT-M perp 1h, 2025-11-01 -> 2026-10-31. Always-in-market, 1x notional.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_lab.data import binance
from quant_lab.engine.costs import CostModel
from quant_lab.results import write_result

START, END = "2025-11-01", "2026-10-31"
K = 5
COSTS = CostModel()


def heikin_ashi(b: pd.DataFrame) -> pd.DataFrame:
    ha_c = (b.open + b.high + b.low + b.close) / 4
    ha_o = np.empty(len(b))
    ha_o[0] = (b.open.iloc[0] + b.close.iloc[0]) / 2
    hc = ha_c.to_numpy()
    for i in range(1, len(b)):
        ha_o[i] = (ha_o[i - 1] + hc[i - 1]) / 2
    return pd.DataFrame({"open": ha_o, "close": hc}, index=b.index)


def trades_from_positions(pos: np.ndarray, entry_px: np.ndarray) -> pd.DataFrame:
    """pos[i] = position held after bar i's decision; fills at entry_px[i] whenever pos changes."""
    rows, cur, px0 = [], 0, None
    for i in range(len(pos)):
        if pos[i] != cur:
            if cur != 0:
                rows.append({"side": cur, "entry": px0, "exit": entry_px[i]})
            cur, px0 = pos[i], entry_px[i]
    t = pd.DataFrame(rows)
    t["gross_ret"] = t.side * (t.exit - t.entry) / t.entry
    return t


def stats(t: pd.DataFrame, cost: float) -> dict:
    net = t.gross_ret - cost
    return {"trades": int(len(t)), "win_rate": float((net > 0).mean()),
            "total_return": float((1 + net).prod() - 1), "win_rate_before_costs": float((t.gross_ret > 0).mean()),
            "total_return_before_costs": float((1 + t.gross_ret).prod() - 1)}


def main():
    b = binance.klines("BTCUSDT", "1h", START, END, market="um")
    rt = COSTS.round_trip("taker", "taker")
    out = {"buy_and_hold": float(b.close.iloc[-1] / b.open.iloc[0] - 1), "round_trip_cost": rt}

    # ---- 1) Heikin-Ashi --------------------------------------------------------------
    ha = heikin_ashi(b)
    sig = np.where(ha.close > ha.open, 1, -1)             # known at close of bar i
    # (a) fake: fill at the HA close of the signal bar (synthetic price, same-bar)
    fake = trades_from_positions(sig, ha.close.to_numpy())
    # (b) real: act at the NEXT bar's real open
    real_pos = np.concatenate([[0], sig[:-1]])
    real = trades_from_positions(real_pos, b.open.to_numpy())
    out["heikin_ashi"] = {"fake_fills_no_fee": stats(fake, 0.0), "real_fills_no_fee": stats(real, 0.0),
                          "real_fills_with_fee": stats(real, rt)}

    # ---- 2) Repainting pivots ----------------------------------------------------------
    h, l = b.high.to_numpy(), b.low.to_numpy()
    n = len(b)
    piv = np.zeros(n, dtype=int)
    for p in range(K, n - K):
        if l[p] == l[p - K:p + K + 1].min():
            piv[p] = 1
        elif h[p] == h[p - K:p + K + 1].max():
            piv[p] = -1
    # (a) repainted history: position switches ON the pivot bar, filled at that bar's extreme
    pos_a, px_a, cur = np.zeros(n, int), np.zeros(n), 0
    for i in range(n):
        if piv[i]:
            cur = piv[i]
        pos_a[i] = cur
        px_a[i] = l[i] if piv[i] == 1 else (h[i] if piv[i] == -1 else b.close.iloc[i])
    repaint = trades_from_positions(pos_a, px_a)
    # (b) live: the pivot is only known at bar p+K; act at the open of bar p+K+1
    pos_b = np.zeros(n, int)
    cur = 0
    for i in range(n):
        p = i - K - 1
        if p >= 0 and piv[p]:
            cur = piv[p]
        pos_b[i] = cur
    live = trades_from_positions(pos_b, b.open.to_numpy())
    out["repainting_pivots"] = {"k": K, "repainted_history": stats(repaint, 0.0),
                                "live_no_fee": stats(live, 0.0), "live_with_fee": stats(live, rt)}

    for sec, d in out.items():
        if isinstance(d, dict):
            for k, s in d.items():
                if isinstance(s, dict):
                    print(f"{sec:18} {k:22} trades {s['trades']:>4} win {s['win_rate']:6.1%} ret {s['total_return']:+9.1%}")
        else:
            print(sec, d)
    write_result("ep01", "fake_backtest_tricks", out,
                 params={"symbol": "BTCUSDT", "interval": "1h", "range": [START, END], "pivot_k": K,
                         "costs": COSTS.as_dict()})


if __name__ == "__main__":
    main()

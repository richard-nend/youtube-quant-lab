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
    rows, cur, px0, i0 = [], 0, None, None
    for i in range(len(pos)):
        if pos[i] != cur:
            if cur != 0:
                # entry_i / exit_i are bookkeeping for the chart samples only (stats ignore them)
                rows.append({"side": cur, "entry": px0, "exit": entry_px[i], "entry_i": i0, "exit_i": i})
            cur, px0, i0 = pos[i], entry_px[i], i
    t = pd.DataFrame(rows)
    t["gross_ret"] = t.side * (t.exit - t.entry) / t.entry
    return t


def stats(t: pd.DataFrame, cost: float) -> dict:
    net = t.gross_ret - cost
    return {"trades": int(len(t)), "win_rate": float((net > 0).mean()),
            "total_return": float((1 + net).prod() - 1), "win_rate_before_costs": float((t.gross_ret > 0).mean()),
            "total_return_before_costs": float((1 + t.gross_ret).prod() - 1)}


CURVE_STEP = 24  # chart samples: one equity point per 24 1h bars (daily)
WINDOW = 40      # bars in the zoomed chart windows


def equity_curve(t: pd.DataFrame, cost: float, n: int) -> list[float]:
    """Equity (start 1.0) after every CURVE_STEP bars: compounded net trade returns realised by then.
    The last point is the final equity, so curve[-1] - 1 == stats(t, cost)['total_return']."""
    net = (t.gross_ret - cost).to_numpy()
    exit_i = t.exit_i.to_numpy()
    eq = np.ones(n)
    acc, j = 1.0, 0
    for i in range(n):
        while j < len(net) and exit_i[j] <= i:
            acc *= 1 + net[j]
            j += 1
        eq[i] = acc
    idx = list(range(0, n, CURVE_STEP))
    if idx[-1] != n - 1:
        idx.append(n - 1)
    return [float(eq[i]) for i in idx]


def ohlc_rows(b: pd.DataFrame, s: int, e: int) -> list[dict]:
    return [{"t": ts.isoformat(), "o": float(r.open), "h": float(r.high), "l": float(r.low), "c": float(r.close)}
            for ts, r in b.iloc[s:e].iterrows()]


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

    # ---- chart samples for the video (no effect on the numbers above) -----------------
    for name, t, c, ref in [("ha_fake", fake, 0.0, out["heikin_ashi"]["fake_fills_no_fee"]),
                            ("ha_real", real, 0.0, out["heikin_ashi"]["real_fills_no_fee"]),
                            ("ha_real_fee", real, rt, out["heikin_ashi"]["real_fills_with_fee"]),
                            ("rp_live", live, 0.0, out["repainting_pivots"]["live_no_fee"]),
                            ("rp_live_fee", live, rt, out["repainting_pivots"]["live_with_fee"])]:
        curve = equity_curve(t, c, n)
        assert abs((curve[-1] - 1) - ref["total_return"]) < 1e-9 * max(1.0, abs(ref["total_return"])), name
        out.setdefault("charts", {"curve_step_bars": CURVE_STEP, "curves": {}})["curves"][name] = curve
    out["charts"]["curve_start"] = b.index[0].isoformat()
    out["charts"]["buy_and_hold_curve"] = [float(b.close.iloc[i] / b.open.iloc[0])
                                           for i in list(range(0, n, CURVE_STEP)) + ([n - 1] if (n - 1) % CURVE_STEP else [])]
    # HA window: WINDOW-bar block (non-overlapping) where HA-price fills beat real next-open fills the most
    hc_ = ha.close.to_numpy()
    fake_ret_bar = np.zeros(n)
    real_ret_bar = np.zeros(n)
    for t, arr in [(fake, fake_ret_bar), (real, real_ret_bar)]:
        for _, r in t.iterrows():
            arr[int(r.exit_i)] += r.gross_ret
    best_s, best_gap = 0, -1e9
    for s in range(0, n - WINDOW, WINDOW):
        gap = fake_ret_bar[s:s + WINDOW].sum() - real_ret_bar[s:s + WINDOW].sum()
        if gap > best_gap:
            best_s, best_gap = s, gap
    s, e = best_s, best_s + WINDOW
    out["charts"]["ha_window"] = {
        "start_index": s,
        "real": ohlc_rows(b, s, e),
        "ha": [{"o": float(ha.open.iloc[i]), "c": float(hc_[i])} for i in range(s, e)],
        # flips inside the window: signal bar i, synthetic fill = HA close of bar i, real fill = open of bar i+1
        "flips": [{"i": int(i - s), "side": int(sig[i]), "fake_px": float(hc_[i]), "real_px": float(b.open.iloc[i + 1])}
                  for i in range(max(s, 1), e - 1) if sig[i] != sig[i - 1]],
        "gross_gap_sum": float(best_gap),
    }
    # Repaint window: block where the repainted-history trades beat the live trades the most
    rp_bar = np.zeros(n)
    lv_bar = np.zeros(n)
    for t, arr in [(repaint, rp_bar), (live, lv_bar)]:
        for _, r in t.iterrows():
            arr[int(r.exit_i)] += r.gross_ret
    best_s, best_gap = 0, -1e9
    for s in range(K + 1, n - WINDOW - K - 1, WINDOW):
        lows = int((piv[s:s + WINDOW - K - 1] == 1).sum())
        if lows < 2:
            continue
        gap = rp_bar[s:s + WINDOW].sum() - lv_bar[s:s + WINDOW].sum()
        if gap > best_gap:
            best_s, best_gap = s, gap
    s, e = best_s, best_s + WINDOW
    out["charts"]["repaint_window"] = {
        "start_index": s,
        "k": K,
        "real": ohlc_rows(b, s, e),
        # pivots fully confirmable inside the window: drawn at bar i (price = low/high), live entry at open of i+K+1
        "pivots": [{"i": int(p - s), "type": int(piv[p]), "drawn_px": float(l[p] if piv[p] == 1 else h[p]),
                    "live_i": int(p + K + 1 - s), "live_px": float(b.open.iloc[p + K + 1])}
                   for p in range(s, e - K - 1) if piv[p]],
        "gross_gap_sum": float(best_gap),
    }

    for sec, d in out.items():
        if sec == "charts":
            continue
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

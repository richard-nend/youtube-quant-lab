"""EP04 — long grid bot in ranges vs trends, and the two proposed safety devices.

Narration claims:
  (a) grids earn steadily in ranges but get stuck with a pile of long fills in a trend
  (b) fix 1: ATR-scaled grid spacing; fix 2: refuse new buys after 3 consecutive fills in the
      same direction (until a sell happens)

Method: BTCUSDT perp 1h, 2024-01-01 -> 2026-10-31, every non-overlapping 30-day window.
Each window starts a fresh long grid at the window's first open: 10 buy levels below,
each buy = 10% of capital, each filled level sells one step higher (maker fees both sides).
Windows are labelled by the window's own close-to-open return:
  range  |r| < 5%   ·  downtrend r <= -10%  ·  uptrend r >= +10%   (others "mixed")
Variants: fixed 1% spacing · ATR spacing (1.0 x 14-day ATR%, frozen at grid start) · fixed + 3-fill lock
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_lab.data import binance
from quant_lab.engine.costs import CostModel
from quant_lab.results import write_result

START, END = "2024-01-01", "2026-10-31"
WINDOW_H = 24 * 30
LEVELS, BUY_FRAC = 10, 0.10
COSTS = CostModel()


def run_grid(b: pd.DataFrame, spacing: float, lock_after: int | None) -> dict:
    p0 = b.open.iloc[0]
    levels = [p0 * (1 - spacing * (k + 1)) for k in range(LEVELS)]
    cash, holdings = 1.0, {}          # level index -> qty
    consecutive_buys, min_equity = 0, 1.0
    fee = COSTS.maker_fee
    for _, bar in b.iterrows():
        # sells first (conservative ordering does not matter much at 1h)
        for k in list(holdings):
            sell_px = levels[k] * (1 + spacing)
            if bar.high >= sell_px:
                cash += holdings.pop(k) * sell_px * (1 - fee)
                consecutive_buys = 0
        for k, lv in enumerate(levels):
            if k in holdings or bar.low > lv:
                continue
            if lock_after is not None and consecutive_buys >= lock_after:
                break
            notional = BUY_FRAC
            if cash < notional:
                break
            cash -= notional
            holdings[k] = notional * (1 - fee) / lv
            consecutive_buys += 1
        eq = cash + sum(q * bar.low for q in holdings.values())
        min_equity = min(min_equity, eq)
    last = b.close.iloc[-1]
    final = cash + sum(q * last for q in holdings.values())
    return {"return": final - 1, "max_drawdown": min_equity - 1, "open_levels_at_end": len(holdings),
            "unrealised": sum(q * last for q in holdings.values()) - BUY_FRAC * len(holdings)}


def main():
    b = binance.klines("BTCUSDT", "1h", START, END, market="um")
    daily = binance.klines("BTCUSDT", "1d", "2023-11-01", END, market="um")   # ATR warm-up
    tr = pd.concat([daily.high - daily.low, (daily.high - daily.close.shift()).abs(),
                    (daily.low - daily.close.shift()).abs()], axis=1).max(axis=1)
    atr_pct = (tr.rolling(14).mean() / daily.close).shift(1)      # known before the day starts

    rows = []
    for s in range(0, len(b) - WINDOW_H, WINDOW_H):
        w = b.iloc[s: s + WINDOW_H]
        r = w.close.iloc[-1] / w.open.iloc[0] - 1
        regime = "range" if abs(r) < 0.05 else "downtrend" if r <= -0.10 else "uptrend" if r >= 0.10 else "mixed"
        a = float(atr_pct.asof(w.index[0]))
        row = {"start": str(w.index[0].date()), "btc_return": r, "regime": regime, "atr_pct": a}
        row["fixed_1pct"] = run_grid(w, 0.01, None)
        row["atr_spacing"] = run_grid(w, max(a, 0.005), None)
        row["fixed_1pct_lock3"] = run_grid(w, 0.01, 3)
        rows.append(row)

    summary = {}
    for regime in ("range", "downtrend", "uptrend", "mixed"):
        rs = [r for r in rows if r["regime"] == regime]
        if not rs:
            continue
        summary[regime] = {"windows": len(rs), "btc_mean": float(np.mean([r["btc_return"] for r in rs]))}
        for v in ("fixed_1pct", "atr_spacing", "fixed_1pct_lock3"):
            summary[regime][v] = {
                "mean_return": float(np.mean([r[v]["return"] for r in rs])),
                "worst_return": float(np.min([r[v]["return"] for r in rs])),
                "mean_max_dd": float(np.mean([r[v]["max_drawdown"] for r in rs])),
                "mean_open_levels_end": float(np.mean([r[v]["open_levels_at_end"] for r in rs])),
            }
    for k, s in summary.items():
        print(f"\n{k} ({s['windows']} windows, BTC mean {s['btc_mean']:+.1%})")
        for v in ("fixed_1pct", "atr_spacing", "fixed_1pct_lock3"):
            x = s[v]
            print(f"  {v:18} mean {x['mean_return']:+6.2%} worst {x['worst_return']:+6.2%} "
                  f"avgDD {x['mean_max_dd']:+6.2%} stuck levels {x['mean_open_levels_end']:.1f}")
    write_result("ep04", "grid_bot_sim", {"summary": summary, "windows": rows},
                 params={"range": [START, END], "window_hours": WINDOW_H, "levels": LEVELS,
                         "buy_frac": BUY_FRAC, "costs": COSTS.as_dict()})


if __name__ == "__main__":
    main()

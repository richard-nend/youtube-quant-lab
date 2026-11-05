"""EP04 — verify the black-swan claims against Binance data.

Narration claims:
  (a) LUNA collapse: "BTC printed 12 consecutive red 15-minute candles"
  (b) FTX collapse:  "BTC fell -24% in just 2 hours"

Method: BTCUSDT spot 15m (streaks) and 1m (worst rolling 2h drop) over each window.
"""

from __future__ import annotations

import pandas as pd

from quant_lab.data import binance
from quant_lab.results import write_result

WINDOWS = {
    "luna_2022_05": ("2022-05-05", "2022-05-15"),
    "ftx_2022_11": ("2022-11-06", "2022-11-12"),
}


def longest_red_streak(bars: pd.DataFrame) -> dict:
    red = (bars["close"] < bars["open"]).to_numpy()
    best, cur, end = 0, 0, None
    for i, r in enumerate(red):
        cur = cur + 1 if r else 0
        if cur > best:
            best, end = cur, i
    start = end - best + 1
    return {"length": best, "start": str(bars.index[start]), "end": str(bars.index[end]),
            "move_pct": float(bars["close"].iloc[end] / bars["open"].iloc[start] - 1)}


def worst_window_drop(bars_1m: pd.DataFrame, minutes: int) -> dict:
    """Worst peak-to-trough inside any rolling window: high at t0 -> lowest low within `minutes`."""
    future_low = bars_1m["low"][::-1].rolling(minutes, min_periods=1).min()[::-1]
    drop = future_low / bars_1m["high"] - 1
    t = drop.idxmin()
    return {"drop_pct": float(drop.min()), "from": str(t), "high": float(bars_1m.loc[t, "high"])}


def main():
    out = {}
    for name, (s, e) in WINDOWS.items():
        b15 = binance.klines("BTCUSDT", "15m", s, e, market="spot")
        b1 = binance.klines("BTCUSDT", "1m", s, e, market="spot")
        out[name] = {
            "longest_red_streak_15m": longest_red_streak(b15),
            "worst_2h_drop": worst_window_drop(b1, 120),
            "worst_24h_drop": worst_window_drop(b1, 1440),
            "window_high_to_low": float(b1["low"].min() / b1["high"].max() - 1),
        }
        print(name, out[name])
    print("wrote", write_result("ep04", "blackswan_scanner", out,
                                params={"symbol": "BTCUSDT", "market": "spot", "windows": WINDOWS}))


if __name__ == "__main__":
    main()

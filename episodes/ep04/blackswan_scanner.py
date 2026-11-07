"""EP04 — verify the black-swan claims against Binance data.

Narration claims:
  (a) LUNA collapse: "BTC printed 12 consecutive red 15-minute candles"
  (b) FTX collapse:  "BTC fell -24% in just 2 hours"

Method: BTCUSDT spot 15m (streaks) and 1m (worst rolling 2h drop) over each window.
"""

from __future__ import annotations

import pandas as pd

from quant_lab.data import binance
from quant_lab.results import ohlc_rows, write_result

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


def charts(b15: pd.DataFrame, r: dict) -> dict:
    """15m candles around the measured events (video only; no effect on the numbers above)."""
    st = r["longest_red_streak_15m"]
    i0, i1 = b15.index.get_loc(pd.Timestamp(st["start"])), b15.index.get_loc(pd.Timestamp(st["end"]))
    assert i1 - i0 + 1 == st["length"]
    pad = 8
    streak = {"streak_start_i": pad if i0 >= pad else i0, "streak_len": st["length"],
              "bars": ohlc_rows(b15, i0 - pad, i1 + 1 + pad)}
    # worst 2h drop: 1m peak at `from`; show the 15m bars from 2h before to 4h after it
    t0 = pd.Timestamp(r["worst_2h_drop"]["from"])
    k = b15.index.searchsorted(t0, side="right") - 1          # 15m bar containing the 1m peak
    s = max(k - 8, 0)
    two_h = {"peak_time_1m": r["worst_2h_drop"]["from"], "peak_high_1m": r["worst_2h_drop"]["high"],
             "drop_pct": r["worst_2h_drop"]["drop_pct"], "peak_bar_i": int(k - s), "window_bars": 8,
             "bars": ohlc_rows(b15, s, k + 17)}
    hourly = b15.resample("1h").agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    return {"red_streak_15m": streak, "worst_2h_15m": two_h, "whole_window_1h": ohlc_rows(hourly, 0, len(hourly))}


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
        out[name]["charts"] = charts(b15, out[name])
    print("wrote", write_result("ep04", "blackswan_scanner", out,
                                params={"symbol": "BTCUSDT", "market": "spot", "windows": WINDOWS}))


if __name__ == "__main__":
    main()

"""EP06 — 2024-12-05, BTC's first break of $100,000.

Narration claims:
  (a) "During the big green candle, open interest actually DECREASED"
      (=> short covering, not new longs)
  (b) "Hours later, late longs were swept by a liquidation cascade"

Method: BTCUSDT USDT-M perpetual 1m klines + 5m open interest (Binance metrics archive).
The impulse = the 2 hours of price action ending at the first 1m bar trading >= $100,000.
"""

from __future__ import annotations

import pandas as pd

from quant_lab.data import binance
from quant_lab.results import write_result

SYMBOL = "BTCUSDT"
DAYS = ["2024-12-04", "2024-12-05", "2024-12-06"]
IMPULSE_HOURS = 2


def oi_at(oi: pd.Series, t: pd.Timestamp) -> float:
    return float(oi[oi.index <= t].iloc[-1])


def main():
    bars = binance.klines(SYMBOL, "1m", DAYS[0], DAYS[-1], market="um")
    oi = pd.concat([binance.futures_metrics(SYMBOL, d) for d in DAYS])["sum_open_interest"].dropna()

    cross = bars.index[bars["high"] >= 100_000][0]
    imp_start = cross - pd.Timedelta(hours=IMPULSE_HOURS)
    imp = bars.loc[imp_start:cross]
    impulse = {
        "first_trade_above_100k": str(cross),
        "price_from": float(imp["open"].iloc[0]),
        "price_to": float(imp["close"].iloc[-1]),
        "price_change_pct": float(imp["close"].iloc[-1] / imp["open"].iloc[0] - 1),
        "oi_from_btc": oi_at(oi, imp_start),
        "oi_to_btc": oi_at(oi, cross),
    }
    impulse["oi_change_pct"] = impulse["oi_to_btc"] / impulse["oi_from_btc"] - 1

    # robustness: the verdict must not depend on how the "big green candle" is delimited
    peak_run = bars.loc[cross: cross + pd.Timedelta(hours=1), "high"].idxmax()
    windows = []
    for start in ["2024-12-04 20:00", "2024-12-05 00:00", str(cross)]:
        a = pd.Timestamp(start, tz="UTC")
        p0, p1 = float(bars.loc[a:, "open"].iloc[0]), float(bars.loc[:peak_run, "close"].iloc[-1])
        o0, o1 = oi_at(oi, a), oi_at(oi, peak_run)
        windows.append({"from": str(a), "to": str(peak_run), "price_change_pct": p1 / p0 - 1,
                        "oi_change_pct": o1 / o0 - 1})
    impulse["windows"] = windows
    impulse["verdict_oi_decreased_during_breakout"] = all(w["oi_change_pct"] < 0 for w in windows)

    # largest drop within 24h after the cross
    after = bars.loc[cross: cross + pd.Timedelta(hours=24)]
    peak_t = after["high"].idxmax()
    trough_t = after.loc[peak_t:]["low"].idxmin()
    drop = {
        "peak_time": str(peak_t), "peak": float(after.loc[peak_t, "high"]),
        "trough_time": str(trough_t), "trough": float(after.loc[trough_t, "low"]),
        "hours_after_cross": (trough_t - cross).total_seconds() / 3600,
    }
    drop["drop_pct"] = drop["trough"] / drop["peak"] - 1
    # fastest leg: worst 30-minute window around the trough
    leg = bars.loc[trough_t - pd.Timedelta(minutes=30): trough_t]
    drop["last_30m_drop_pct"] = float(leg["low"].min() / leg["high"].max() - 1)
    drop["oi_before_drop_btc"] = oi_at(oi, trough_t - pd.Timedelta(minutes=30))
    drop["oi_after_drop_btc"] = oi_at(oi, trough_t + pd.Timedelta(minutes=10))
    drop["oi_change_pct"] = drop["oi_after_drop_btc"] / drop["oi_before_drop_btc"] - 1

    # 5-minute series around the event for the on-screen chart
    win = slice(cross - pd.Timedelta(hours=6), trough_t + pd.Timedelta(hours=2))
    px5 = bars.loc[win, "close"].resample("5min").last()
    series = pd.DataFrame({"close": px5, "oi_btc": oi.loc[win].reindex(px5.index, method="ffill")}).dropna()

    values = {"impulse": impulse, "post_cross_drop": drop,
              "series_5m": [{"t": str(t), "close": r.close, "oi_btc": r.oi_btc} for t, r in series.iterrows()]}
    print("impulse:", impulse)
    print("drop:", drop)
    print("wrote", write_result("ep06", "dec2024_oi_autopsy", values,
                                params={"symbol": SYMBOL, "market": "um", "impulse_hours": IMPULSE_HOURS}))


if __name__ == "__main__":
    main()

"""EP02 — find every "BTC drops >= 2% from ~$68,000 within 30 minutes" event, by rule.

Pre-registered trigger (uses only information up to the trigger minute):
  - some 1m high in the prior 30 minutes is within [67,000, 69,000]
  - the current 1m close is >= 2% below that high
  - events closer than 24h to a previous event are merged (first one kept)
Every event that fires is used for the replay — no selection on what happened next.
"""

from __future__ import annotations

import pandas as pd

from quant_lab.data import binance
from quant_lab.results import write_result

BAND = (67_000, 69_000)
DROP = 0.02
LOOKBACK_MIN = 30
START, END = "2024-01-01", "2026-10-31"


def find_events(bars: pd.DataFrame) -> pd.DataFrame:
    hi = bars["high"].where(bars["high"].between(*BAND))
    ref_high = hi.rolling(LOOKBACK_MIN, min_periods=1).max().shift(1)
    trig = bars["close"] <= ref_high * (1 - DROP)
    times = bars.index[trig.fillna(False).to_numpy()]
    events, last = [], None
    for t in times:
        if last is None or t - last >= pd.Timedelta(hours=24):
            after = bars.loc[t: t + pd.Timedelta(minutes=60)]
            events.append({
                "trigger_time": str(t),
                "ref_high": float(ref_high.loc[t]),
                "trigger_close": float(bars.loc[t, "close"]),
                # descriptive only (NOT used for selection):
                "next60_low_pct": float(after["low"].min() / bars.loc[t, "close"] - 1),
                "next60_close_pct": float(after["close"].iloc[-1] / bars.loc[t, "close"] - 1),
            })
            last = t
    return pd.DataFrame(events)


def main():
    bars = binance.klines("BTCUSDT", "1m", START, END, market="um")
    ev = find_events(bars)
    print(ev.to_string())
    write_result("ep02", "crash_events", {"events": ev.to_dict("records")},
                 params={"band": BAND, "drop": DROP, "lookback_min": LOOKBACK_MIN, "range": [START, END]},
                 sources=[f"data.binance.vision futures/um monthly klines BTCUSDT 1m {START}..{END}"])


if __name__ == "__main__":
    main()

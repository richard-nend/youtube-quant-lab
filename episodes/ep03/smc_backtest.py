"""EP03 — SMC (FVG / order block) backtest on BTCUSDT perp 15m, one year.

Narration claims:
  (a) mechanical entry on every FVG and OB with the `smartmoneyconcepts` package -> win rate 34.2%
  (b) filters (sweep first, displacement body>=75% & range>=1.8xATR, discount zone, London/NY
      kill zones, CVD absorption divergence) -> trades -70%, win 52%, R:R 1:3.2, positive expectancy
  (c) shorts: R:R 1:4

Variants (all share entry/exit rules; thresholds are taken from the narration, not tuned):
  A naive_lib   : smartmoneyconcepts fvg()+ob() outputs used at their own index.
                  LOOK-AHEAD: fvg() reads the next candle; swing_highs_lows() reads swing_length/2
                  future candles; ob() deletes order blocks that later fail (survivorship).
  B causal_all  : same idea, rebuilt causally (every signal only usable after the bar that confirms it).
  C1..C5        : B's FVGs + narration filters added cumulatively.

Trade rules: limit at the near edge of the zone, stop at the far edge of the displacement
candle (FVG) / OB, target 3R, limit valid 20 bars, max hold 96 bars (1 day),
fill="through" (price must trade beyond the limit), costs = Binance VIP0.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

from quant_lab.data import binance
from quant_lab.engine.backtest import Signal, simulate
from quant_lab.engine.costs import CostModel
from quant_lab.metrics import summarize
from quant_lab.results import write_result

START, END = "2025-11-01", "2026-10-31"
RR = 3.0
SWING = 10            # pivot half-window (bars on each side) for swings
ATR_N = 14
DISP_ATR = 1.8
BODY_MIN = 0.75
RANGE_BARS = 96       # dealing range for premium/discount (1 day of 15m bars)
SWEEP_LOOKBACK = 10
CVD_WIN = 20
KILLZONES = [(6.0, 10.0), (11.5, 15.5)]   # UTC hours: London open, NY open (±2h)
COSTS = CostModel()


def atr(b: pd.DataFrame, n: int = ATR_N) -> pd.Series:
    tr = pd.concat([b.high - b.low, (b.high - b.close.shift()).abs(), (b.low - b.close.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


# ---------------------------------------------------------------- causal detectors
def confirmed_swings(b: pd.DataFrame, k: int = SWING):
    """Pivot at bar p is known only at bar p+k. Returns arrays of (confirm_bar -> pivot level)."""
    h, l = b.high.to_numpy(), b.low.to_numpy()
    n = len(b)
    last_sh = np.full(n, np.nan)
    last_sl = np.full(n, np.nan)
    last_sl_bar = np.full(n, -1)
    sh = sl = np.nan
    sl_bar = -1
    for i in range(n):
        p = i - k
        if p - k >= 0:
            if h[p] == h[p - k:i + 1].max():
                sh = h[p]
            if l[p] == l[p - k:i + 1].min():
                sl, sl_bar = l[p], p
        last_sh[i], last_sl[i], last_sl_bar[i] = sh, sl, sl_bar
    return last_sh, last_sl, last_sl_bar


def causal_fvgs(b: pd.DataFrame):
    """Bullish FVG known at close of bar i: high[i-2] < low[i]. Displacement candle = i-1."""
    h, l = b.high.to_numpy(), b.low.to_numpy()
    out = []
    for i in range(2, len(b)):
        if h[i - 2] < l[i]:
            out.append({"bar": i, "side": 1, "edge": l[i], "far": h[i - 2], "disp": i - 1})
        elif l[i - 2] > h[i]:
            out.append({"bar": i, "side": -1, "edge": h[i], "far": l[i - 2], "disp": i - 1})
    return out


def causal_obs(b: pd.DataFrame, last_sh, last_sl):
    """OB known at the break-of-structure bar: close beyond the last CONFIRMED swing.
    Bullish OB = last down candle before the BOS bar (within 20 bars)."""
    o, c, h, l = (b[x].to_numpy() for x in ("open", "close", "high", "low"))
    out, used_h, used_l = [], set(), set()
    for i in range(1, len(b)):
        if not np.isnan(last_sh[i]) and c[i] > last_sh[i] and last_sh[i] not in used_h:
            used_h.add(last_sh[i])
            for j in range(i - 1, max(i - 20, 0), -1):
                if c[j] < o[j]:
                    out.append({"bar": i, "side": 1, "edge": h[j], "far": l[j]})
                    break
        if not np.isnan(last_sl[i]) and c[i] < last_sl[i] and last_sl[i] not in used_l:
            used_l.add(last_sl[i])
            for j in range(i - 1, max(i - 20, 0), -1):
                if c[j] > o[j]:
                    out.append({"bar": i, "side": -1, "edge": l[j], "far": h[j]})
                    break
    return out


def to_signal(z: dict, stop: float) -> Signal | None:
    entry = z["edge"]
    risk = (entry - stop) * z["side"]
    if risk <= 0:
        return None
    return Signal(bar=z["bar"], side=z["side"], stop=stop, target=entry + z["side"] * RR * risk,
                  entry="limit", limit_price=entry)


def fvg_signal(z, l, h):
    stop = l[z["disp"]] if z["side"] > 0 else h[z["disp"]]
    return to_signal(z, stop)


# ---------------------------------------------------------------- variant A: library, naive
def naive_lib_signals(b: pd.DataFrame):
    sys_stdout = sys.stdout
    sys.stdout = open(os.devnull, "w")          # silence the package's banner
    try:
        from smartmoneyconcepts import smc
    finally:
        sys.stdout = sys_stdout
    ohlc = b[["open", "high", "low", "close", "volume"]].reset_index(drop=True)
    f = smc.fvg(ohlc)
    sw = smc.swing_highs_lows(ohlc, swing_length=2 * SWING)
    ob = smc.ob(ohlc, sw)
    l, h = b.low.to_numpy(), b.high.to_numpy()
    sigs = []
    for i in np.flatnonzero(f["FVG"].fillna(0).to_numpy() != 0):
        side = int(f["FVG"].iloc[i])
        z = {"bar": int(i), "side": side, "edge": f["Top"].iloc[i] if side > 0 else f["Bottom"].iloc[i], "disp": int(i)}
        s = fvg_signal(z, l, h)
        if s:
            sigs.append(s)
    for i in np.flatnonzero(ob["OB"].fillna(0).to_numpy() != 0):
        side = int(ob["OB"].iloc[i])
        edge, far = (ob["Top"].iloc[i], ob["Bottom"].iloc[i]) if side > 0 else (ob["Bottom"].iloc[i], ob["Top"].iloc[i])
        s = to_signal({"bar": int(i), "side": side, "edge": edge}, far)
        if s:
            sigs.append(s)
    return sigs


# ---------------------------------------------------------------- run
def run(b: pd.DataFrame, sigs) -> dict:
    t = simulate(b, sigs, COSTS, fill="through", limit_ttl=20, max_hold=96)
    s = summarize(t, position_frac=0.10)   # 10% of equity notional per trade, for the equity curve
    if not t.empty:
        s["avg_r_win"] = float(t.loc[t.net_ret > 0, "r_multiple"].mean()) if (t.net_ret > 0).any() else None
        s["avg_r_loss"] = float(t.loc[t.net_ret <= 0, "r_multiple"].mean()) if (t.net_ret <= 0).any() else None
        s["expectancy_r"] = float(t["r_multiple"].mean())
        s["signals"] = len(sigs)
        m = t.set_index("entry_time")["net_ret"].resample("MS")
        s["monthly"] = {str(k.date()): {"trades": int(v.count()), "win_rate": float((v > 0).mean()) if v.count() else None}
                        for k, v in m}
    return s


def main():
    b = binance.klines("BTCUSDT", "15m", START, END, market="um")
    l, h, o, c = b.low.to_numpy(), b.high.to_numpy(), b.open.to_numpy(), b.close.to_numpy()
    a = atr(b).to_numpy()
    delta = (2 * b.taker_buy_volume - b.volume).to_numpy()          # taker buy - taker sell, per bar
    cvd = np.cumsum(delta)
    last_sh, last_sl, last_sl_bar = confirmed_swings(b)
    rng_hi = b.high.rolling(RANGE_BARS).max().to_numpy()
    rng_lo = b.low.rolling(RANGE_BARS).min().to_numpy()
    hours = (b.index.hour + b.index.minute / 60).to_numpy()

    fvgs = causal_fvgs(b)
    obs = causal_obs(b, last_sh, last_sl)

    # --- filter predicates (evaluated only with information available at z["bar"]) ---
    def swept(z):
        """a sweep of the last confirmed swing low (long) / high (short) within the lookback"""
        i = z["bar"]
        for j in range(max(i - SWEEP_LOOKBACK, 1), i + 1):
            if z["side"] > 0 and not np.isnan(last_sl[j - 1]) and l[j] < last_sl[j - 1] < c[j]:
                return j
            if z["side"] < 0 and not np.isnan(last_sh[j - 1]) and h[j] > last_sh[j - 1] > c[j]:
                return j
        return None

    def displacement(z):
        d = z["disp"]
        rng = h[d] - l[d]
        return rng > 0 and abs(c[d] - o[d]) / rng >= BODY_MIN and rng >= DISP_ATR * a[d] and np.sign(c[d] - o[d]) == z["side"]

    def discount(z):
        i = z["bar"]
        mid = (rng_hi[i] + rng_lo[i]) / 2
        return (z["edge"] < mid) if z["side"] > 0 else (z["edge"] > mid)

    def killzone(z):
        hr = hours[z["bar"]]
        return any(a0 <= hr < a1 for a0, a1 in KILLZONES)

    def absorption(z):
        """price swept below the prior swing low while CVD stayed ABOVE its value at that swing low"""
        j = swept(z)
        if j is None:
            return False
        ref = int(last_sl_bar[j - 1]) if z["side"] > 0 else None
        if z["side"] > 0:
            return ref >= 0 and cvd[j] > cvd[ref]
        # bearish mirror: swept above swing high while CVD stayed below its value there (approximate ref = j-CVD_WIN)
        return cvd[j] < cvd[max(j - CVD_WIN, 0)]

    steps = [
        ("C1_sweep", lambda z: swept(z) is not None),
        ("C2_+displacement", displacement),
        ("C3_+discount", discount),
        ("C4_+killzone", killzone),
        ("C5_+cvd_absorption", absorption),
    ]

    results = {}
    results["A_naive_lib"] = run(b, naive_lib_signals(b))
    all_causal = [s for s in (fvg_signal(z, l, h) for z in fvgs) if s] + \
                 [s for s in (to_signal(z, z["far"]) for z in obs) if s]
    results["B_causal_all"] = run(b, all_causal)
    results["B_causal_fvg_only"] = run(b, [s for s in (fvg_signal(z, l, h) for z in fvgs) if s])

    pool = fvgs
    for name, pred in steps:
        pool = [z for z in pool if pred(z)]
        results[name] = run(b, [s for s in (fvg_signal(z, l, h) for z in pool) if s])

    print(f"{'variant':22} {'trades':>6} {'win':>6} {'avgRwin':>7} {'expR':>6} {'ret(10%)':>9} {'preCost':>8} {'maxDD':>7}")
    for k, s in results.items():
        if s.get("trades", 0) == 0:
            print(f"{k:22} {0:>6}")
            continue
        print(f"{k:22} {s['trades']:>6} {s['win_rate']:>6.1%} {s['avg_r_win'] or 0:>7.2f} {s['expectancy_r']:>6.2f} "
              f"{s['total_return']:>9.1%} {s['total_return_before_costs']:>8.1%} {s['max_drawdown']:>7.1%}")

    write_result("ep03", "smc_backtest", results,
                 params={"symbol": "BTCUSDT", "interval": "15m", "range": [START, END], "rr": RR, "swing": SWING,
                         "disp_atr": DISP_ATR, "body_min": BODY_MIN, "range_bars": RANGE_BARS,
                         "killzones_utc": KILLZONES, "fill": "through", "position_frac": 0.10,
                         "costs": COSTS.as_dict()})


if __name__ == "__main__":
    main()

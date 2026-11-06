"""EP06 answer experiment — H-EP06 in PREREGISTRATION.md (order flow as a confirmation filter).

1h Donchian(20) breakout, stop 2 ATR, target 4 ATR, max hold 48 bars, market entry next open.
  T0 all breakouts · T1 breakout-bar delta z >= +1 in the breakout direction · T2 delta z <= -1 against it.
Delta = 2*taker_buy_volume - volume; z over the previous 720 bars (30 days, past only).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_lab.data import binance
from quant_lab.engine.backtest import Signal
from quant_lab.engine.costs import CostModel
from quant_lab.protocol import run_sets, verdict
from quant_lab.results import write_result

COSTS = CostModel()
DON, ZWIN, Z_TH = 20, 720, 1.0
SL_ATR, TP_ATR, HOLD = 2.0, 4.0, 48
SIM = {"max_hold": HOLD}
SETS = {"BTC-D": ("BTCUSDT", "2025-11-01", "2026-10-31"),
        "BTC-H": ("BTCUSDT", "2024-11-01", "2025-10-31"),
        "ETH-H": ("ETHUSDT", "2024-11-01", "2026-10-31")}


def atr(b, n=14):
    tr = pd.concat([b.high - b.low, (b.high - b.close.shift()).abs(), (b.low - b.close.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def breakout_signals(b: pd.DataFrame, mode: str) -> list[Signal]:
    c = b.close.to_numpy()
    a = atr(b).to_numpy()
    up = b.high.rolling(DON).max().shift(1).to_numpy()     # previous 20 bars only
    dn = b.low.rolling(DON).min().shift(1).to_numpy()
    delta = 2 * b.taker_buy_volume - b.volume
    past = delta.shift(1).rolling(ZWIN)
    z = ((delta - past.mean()) / past.std()).to_numpy()
    sigs = []
    for i in range(ZWIN + 1, len(b)):
        side = 1 if c[i] > up[i] else (-1 if c[i] < dn[i] else 0)
        if side == 0 or np.isnan(z[i]):
            continue
        if mode == "T1" and side * z[i] < Z_TH:
            continue
        if mode == "T2" and side * z[i] > -Z_TH:
            continue
        sigs.append(Signal(bar=i, side=side, stop=c[i] - side * SL_ATR * a[i], target=c[i] + side * TP_ATR * a[i]))
    return sigs


def main():
    # z needs 30 days of history before each period starts
    sets = {}
    for k, (sym, s, e) in SETS.items():
        warm = (pd.Timestamp(s) - pd.Timedelta(days=31)).date()
        b = binance.klines(sym, "1h", warm, e, market="um")
        sets[k] = b
    out = {}
    for mode in ("T0", "T1", "T2"):
        def make(b, mode=mode):
            # signals only inside the period (warm-up bars excluded)
            k0 = next(k for k, v in sets.items() if v is b)
            start = pd.Timestamp(SETS[k0][1], tz=b.index.tz)
            return [s for s in breakout_signals(b, mode) if b.index[s.bar] >= start]
        res = run_sets(make, sets, COSTS, SIM)
        res["verdict"] = verdict(res)
        out[mode] = res
        print(f"{mode} {res['verdict']:5} " + "  ".join(
            f"{k}: n={v['trades']:>4} win={v.get('win_rate', 0):.1%} exp={v.get('expectancy', 0):+.3%} "
            f"pre={v.get('expectancy_before_costs', 0):+.3%} ci=[{v.get('ci95', [0, 0])[0]:+.3%},{v.get('ci95', [0, 0])[1]:+.3%}]"
            for k, v in res.items() if k != "verdict"), flush=True)
    write_result("ep06", "flow_confirmed_breakout", out,
                 params={"sets": SETS, "donchian": DON, "z_window": ZWIN, "z_threshold": Z_TH, "sl_atr": SL_ATR,
                         "tp_atr": TP_ATR, "max_hold": HOLD, "costs": COSTS.as_dict(),
                         "preregistration": "PREREGISTRATION.md H-EP06 (commit 197476e)"})


if __name__ == "__main__":
    main()

"""EP07B L1 split (Amendment 3): the 36-month walk-forward shown as two periods.

unseen   2023-11..2024-10  — before any candidate (EP03 FVG, EP06 flow) was designed
overlap  2024-11..2026-10  — the data the candidates were built on
Same procedure as assembly_walkforward.py L1; only the reporting is split.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd

from quant_lab import LAB_ROOT
from quant_lab.data import binance
from quant_lab.engine.backtest import simulate
from quant_lab.protocol import trade_stats
from quant_lab.results import write_result

_spec = importlib.util.spec_from_file_location("assembly", LAB_ROOT / "episodes/ep07/assembly_walkforward.py")
asm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(asm)

SPLIT = pd.Timestamp("2024-11-01", tz="UTC")


def l1_trades(sym):
    bars = {tf: binance.klines(sym, tf, asm.START, asm.END, market="um") for tf in ("1h", "4h")}
    trades = {}
    for name, (tf, zwin, fn, sim) in asm.CANDIDATES.items():
        b = bars[tf]
        trades[name] = simulate(b, fn(b) if fn else asm.flow_signals(b, zwin), asm.COSTS, **sim)
    out = []
    for m in asm.MONTHS:
        score = {n: (float(w.net_ret.sum()) if len(w := asm.month_slice(t, m - pd.DateOffset(months=12), m)) >= 30 else -np.inf)
                 for n, t in trades.items()}
        best = max(score, key=score.get)
        if score[best] > -np.inf:
            out.append(asm.month_slice(trades[best], m, m + pd.DateOffset(months=1)))
    return pd.concat(out, ignore_index=True).sort_values("exit_time").reset_index(drop=True)


def main():
    res = {}
    for sym in ("BTCUSDT", "ETHUSDT"):
        t = l1_trades(sym)
        res[sym] = {"all": trade_stats(t) | {k: v for k, v in asm.curve_stats(t).items() if k != "curve"}}
        for tag, part in (("unseen_2023-11_2024-10", t[t.signal_time < SPLIT]), ("overlap_2024-11_2026-10", t[t.signal_time >= SPLIT])):
            res[sym][tag] = trade_stats(part) | {k: v for k, v in asm.curve_stats(part).items() if k != "curve"}
        # self-check: the split covers every L1 trade
        assert res[sym]["all"]["trades"] == sum(res[sym][k]["trades"] for k in res[sym] if k != "all")
        for k, v in res[sym].items():
            print(sym, k, v["trades"], round(v["expectancy"], 5), [round(x, 5) for x in v["ci95"]], round(v["final"], 3), round(v["max_dd"], 3), flush=True)
    write_result("ep07", "assembly_l1_split", res,
                 params={"split": str(SPLIT.date()), "source": "assembly_walkforward.py L1 (commit fc415fd)",
                         "preregistration": "PREREGISTRATION.md Amendment 3 (a89b529)"})


if __name__ == "__main__":
    main()

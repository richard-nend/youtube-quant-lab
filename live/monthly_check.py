"""Monthly live check (PREREGISTRATION.md Amendment 3, "Live objective").

For each symbol and month: the live per-trade mean, win rate and trade frequency against the backtest's 95% range
for the same number of trades (L1 trades over the 36 months, resampled 10,000 times, seed 20261107).
Late records are reported alongside, not removed.

    uv run python live/monthly_check.py 2026-11
"""

from __future__ import annotations

import importlib.util
import json
import sys

import numpy as np
import pandas as pd

from quant_lab import LAB_ROOT

N_BOOT, SEED = 10_000, 20261107
LIVE = LAB_ROOT / "live"

_spec = importlib.util.spec_from_file_location("l1_split", LAB_ROOT / "episodes/ep07/l1_split.py")
l1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(l1)


def band(x):
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


def check(live_net, bt_net, bt_monthly_counts, live_days, rng):
    n = len(live_net)
    out = {"trades": n}
    # frequency: trades per 30 days vs the 36 backtest months (resampled)
    freq = n / live_days * 30
    fb = band(rng.choice(bt_monthly_counts, size=N_BOOT))
    out["freq_per_30d"] = {"live": freq, "range95": fb, "within": fb[0] <= freq <= fb[1]}
    if n == 0:
        return out
    draws = rng.choice(bt_net, size=(N_BOOT, n))
    for name, live, sim in (("mean_net", float(np.mean(live_net)), draws.mean(axis=1)),
                            ("win_rate", float(np.mean(live_net > 0)), (draws > 0).mean(axis=1))):
        b = band(sim)
        out[name] = {"live": live, "range95": b, "within": b[0] <= live <= b[1]}
    return out


def main(month: str):
    m0 = pd.Timestamp(month + "-01", tz="UTC")
    m1 = m0 + pd.DateOffset(months=1)
    start = pd.Timestamp(json.loads((LIVE / "start.json").read_text())["start_utc"])
    status = json.loads((LIVE / "status.json").read_text())
    run = pd.Timestamp(status["run_utc"])
    lo, hi = max(m0, start), min(m1, run)
    days = (hi - lo).total_seconds() / 86400

    t = pd.read_csv(LIVE / "trades.csv", parse_dates=["signal_time", "exit_time"])
    t = t[(t.signal_time >= lo) & (t.signal_time < hi)]
    closed = t[~t.open]
    rng = np.random.default_rng(SEED)
    res = {"month": month, "window_utc": [lo.isoformat(), hi.isoformat()], "days": round(days, 2),
           "open_trades": int(t.open.sum()), "late_trades": int(t.late.sum()), "symbols": {}}
    for sym in ("BTCUSDT", "ETHUSDT"):
        bt = l1.l1_trades(sym)
        counts = bt.groupby(bt.signal_time.dt.strftime("%Y-%m")).size().to_numpy()
        c = closed[closed.symbol == sym]
        res["symbols"][sym] = check(c.net_ret.to_numpy(), bt.net_ret.to_numpy(), counts, days, rng)
        res["symbols"][sym]["on_time_only"] = check(c[~c.late].net_ret.to_numpy(), bt.net_ret.to_numpy(), counts, days, rng)
    verdicts = [v["within"] for s in res["symbols"].values() for k, v in s.items() if isinstance(v, dict) and "within" in v]
    res["verdict"] = "within range" if all(verdicts) else "outside range"
    (LIVE / "monthly").mkdir(exist_ok=True)
    (LIVE / "monthly" / f"{month}.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])

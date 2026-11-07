"""Bar-based bracket-order simulator with explicit, conservative fill rules.

Rules (documented because every one of them changes the result):
- Market entries fill at the OPEN of the bar after the signal bar (no same-bar look-ahead).
- Limit entries:
    fill="touch"   -> filled if the bar trades AT the limit price (optimistic)
    fill="through" -> filled only if the bar trades BEYOND the limit by ``through_ticks``
                      (models queue position / adverse selection, EP06)
- If stop and target are both inside one bar, the STOP is assumed first.
- On the bar a LIMIT order fills, only the stop is checked: the bar's high/low may have come
  before the fill, so a same-bar target would be look-ahead.
- If a bar opens beyond the stop (including the entry bar), the exit is at that worse open.
- Unfilled limit orders expire after ``limit_ttl`` bars.
- Positions still open after ``max_hold`` bars exit at that bar's close (taker).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quant_lab.engine.costs import CostModel


@dataclass
class Signal:
    bar: int                 # integer position of the bar on which the signal is known (closed)
    side: int                # +1 long, -1 short
    stop: float
    target: float
    entry: str = "market"    # "market" | "limit"
    limit_price: float | None = None


def simulate(
    bars: pd.DataFrame,
    signals: list[Signal],
    costs: CostModel,
    *,
    fill: str = "through",
    through_ticks: float = 0.0,
    limit_ttl: int = 10,
    max_hold: int = 96,
    allow_overlap: bool = False,
) -> pd.DataFrame:
    o = bars["open"].to_numpy()
    h = bars["high"].to_numpy()
    l = bars["low"].to_numpy()
    c = bars["close"].to_numpy()
    idx = bars.index
    n = len(bars)
    rows = []
    busy_until = -1

    for s in sorted(signals, key=lambda x: x.bar):
        if not allow_overlap and s.bar <= busy_until:
            continue
        start = s.bar + 1
        if start >= n:
            continue

        # ---- entry -------------------------------------------------------
        if s.entry == "market":
            fill_bar, entry_px, entry_type = start, o[start], "taker"
        else:
            lp = s.limit_price
            fill_bar = None
            for j in range(start, min(start + limit_ttl, n)):
                if s.side > 0:
                    hit = l[j] < lp - through_ticks if fill == "through" else l[j] <= lp
                else:
                    hit = h[j] > lp + through_ticks if fill == "through" else h[j] >= lp
                if hit:
                    fill_bar = j
                    break
            if fill_bar is None:
                continue
            entry_px, entry_type = lp, "maker"

        # ---- exit --------------------------------------------------------
        exit_bar, exit_px, exit_type, reason = None, None, None, None
        last = min(fill_bar + max_hold, n - 1)
        for j in range(fill_bar, last + 1):
            if s.side > 0:
                stop_hit, tgt_hit = l[j] <= s.stop, h[j] >= s.target
            else:
                stop_hit, tgt_hit = h[j] >= s.stop, l[j] <= s.target
            if entry_type == "maker" and j == fill_bar:
                tgt_hit = False
            if stop_hit:
                # a gap through the stop fills at the worse open
                gap = (o[j] < s.stop) if s.side > 0 else (o[j] > s.stop)
                exit_px = o[j] if gap else s.stop
                exit_bar, exit_type, reason = j, "taker", "stop"
                break
            if tgt_hit:
                exit_bar, exit_px, exit_type, reason = j, s.target, "maker", "target"
                break
        if exit_bar is None:
            exit_bar, exit_px, exit_type, reason = last, c[last], "taker", "time"

        gross = s.side * (exit_px - entry_px) / entry_px
        cost = costs.side_cost(entry_type) + costs.side_cost(exit_type)
        risk = abs(entry_px - s.stop) / entry_px
        rows.append({
            "signal_time": idx[s.bar],
            "entry_time": idx[fill_bar],
            "exit_time": idx[exit_bar],
            "side": s.side,
            "entry_px": entry_px,
            "exit_px": exit_px,
            "entry_type": entry_type,
            "exit_reason": reason,
            "gross_ret": gross,
            "cost": cost,
            "net_ret": gross - cost,
            "risk": risk,
            "r_multiple": (gross - cost) / risk if risk > 0 else np.nan,
        })
        busy_until = exit_bar

    return pd.DataFrame(rows)

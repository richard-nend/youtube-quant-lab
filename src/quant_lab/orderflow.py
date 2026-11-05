"""Order-flow primitives (EP03, EP06).

Binance aggTrades field ``is_buyer_maker``:
    True  -> the buyer's order was resting in the book (maker); the SELLER crossed
             the spread. This is an aggressive SELL  -> signed volume NEGATIVE.
    False -> the buyer crossed the spread. Aggressive BUY -> signed volume POSITIVE.
Flipping this sign (a common open-source bug) inverts every CVD signal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def signed_volume(trades: pd.DataFrame) -> pd.Series:
    sign = np.where(trades["is_buyer_maker"].to_numpy(), -1.0, 1.0)
    return pd.Series(sign * trades["quantity"].to_numpy(), index=trades.index, name="signed_qty")


def cvd(trades: pd.DataFrame, freq: str | None = None) -> pd.Series:
    """Cumulative volume delta. If ``freq`` is given, resample deltas to bars first."""
    sv = signed_volume(trades)
    if freq:
        sv = sv.resample(freq).sum()
    return sv.cumsum().rename("cvd")


def bar_delta(trades: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Per-bar taker buy volume, taker sell volume and delta."""
    sv = signed_volume(trades)
    buy = sv.clip(lower=0).resample(freq).sum()
    sell = (-sv.clip(upper=0)).resample(freq).sum()
    return pd.DataFrame({"taker_buy": buy, "taker_sell": sell, "delta": buy - sell})

"""Loader for the public Binance archive (https://data.binance.vision).

Every download is cached under quant_lab/.cache/binance/ with the same path as the
remote file, and every loader records the URLs it used in ``SOURCES`` so result
files can cite exactly which raw files produced a number.

Markets:
    "spot" -> data/spot/...
    "um"   -> data/futures/um/...   (USDT-margined perpetuals)
"""

from __future__ import annotations

import io
import zipfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

from quant_lab import CACHE_DIR

BASE_URL = "https://data.binance.vision/data"
SOURCES: list[str] = []  # URLs used during this process, in load order

KLINE_COLS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume", "ignore",
]
AGG_TRADE_COLS_UM = [
    "agg_trade_id", "price", "quantity", "first_trade_id", "last_trade_id",
    "transact_time", "is_buyer_maker",
]
AGG_TRADE_COLS_SPOT = AGG_TRADE_COLS_UM + ["is_best_match"]
METRICS_COLS = [
    "create_time", "symbol", "sum_open_interest", "sum_open_interest_value",
    "count_toptrader_long_short_ratio", "sum_toptrader_long_short_ratio",
    "count_long_short_ratio", "sum_taker_long_short_vol_ratio",
]


def _market_prefix(market: str) -> str:
    if market == "spot":
        return "spot"
    if market == "um":
        return "futures/um"
    raise ValueError(f"unknown market {market!r}")


def _fetch(rel_path: str) -> Path:
    """Download BASE_URL/rel_path into the cache (once) and return the local path."""
    local = CACHE_DIR / "binance" / rel_path
    url = f"{BASE_URL}/{rel_path}"
    if not local.exists():
        local.parent.mkdir(parents=True, exist_ok=True)
        resp = requests.get(url, timeout=120)
        if resp.status_code == 404:
            raise FileNotFoundError(url)
        resp.raise_for_status()
        tmp = local.with_suffix(".part")
        tmp.write_bytes(resp.content)
        tmp.rename(local)
    if url not in SOURCES:
        SOURCES.append(url)
    return local


def _read_zip_csv(path: Path, columns: list[str]) -> pd.DataFrame:
    with zipfile.ZipFile(path) as zf:
        raw = zf.read(zf.namelist()[0])
    first = raw[:1]
    has_header = not first.isdigit()
    df = pd.read_csv(io.BytesIO(raw), header=0 if has_header else None)
    df.columns = columns[: len(df.columns)]
    return df


def _to_utc(ts: pd.Series) -> pd.Series:
    """Binance switched spot files to microseconds in 2025; normalise both units."""
    ts = pd.to_numeric(ts)
    unit = "us" if ts.iloc[0] > 10**14 else "ms"
    return pd.to_datetime(ts, unit=unit, utc=True)


def _months(start: date, end: date):
    cur = date(start.year, start.month, 1)
    while cur <= end:
        nxt = date(cur.year + (cur.month == 12), cur.month % 12 + 1, 1)
        yield cur, nxt - timedelta(days=1)
        cur = nxt


def _days(start: date, end: date):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def _as_date(x) -> date:
    if isinstance(x, datetime):
        return x.date()
    if isinstance(x, date):
        return x
    return date.fromisoformat(str(x)[:10])


def klines(symbol: str, interval: str, start, end, market: str = "um") -> pd.DataFrame:
    """OHLCV bars for [start, end] (inclusive, UTC dates), indexed by open time.

    Whole past months come from monthly archives; partial months fall back to daily files.
    """
    start_d, end_d = _as_date(start), _as_date(end)
    prefix = _market_prefix(market)
    this_month = date.today().replace(day=1)
    frames = []
    for m_start, m_end in _months(start_d, end_d):
        whole = m_start >= start_d.replace(day=1) and m_end <= end_d and m_start < this_month
        if whole:
            tag = m_start.strftime("%Y-%m")
            rel = f"{prefix}/monthly/klines/{symbol}/{interval}/{symbol}-{interval}-{tag}.zip"
            frames.append(_read_zip_csv(_fetch(rel), KLINE_COLS))
        else:
            for d in _days(max(m_start, start_d), min(m_end, end_d)):
                rel = f"{prefix}/daily/klines/{symbol}/{interval}/{symbol}-{interval}-{d.isoformat()}.zip"
                frames.append(_read_zip_csv(_fetch(rel), KLINE_COLS))
    df = pd.concat(frames, ignore_index=True)
    df["open_time"] = _to_utc(df["open_time"])
    df["close_time"] = _to_utc(df["close_time"])
    for c in ["open", "high", "low", "close", "volume", "quote_volume",
              "taker_buy_volume", "taker_buy_quote_volume"]:
        df[c] = df[c].astype(float)
    df = df.drop(columns="ignore").drop_duplicates("open_time").set_index("open_time").sort_index()
    lo = pd.Timestamp(start_d, tz="UTC")
    hi = pd.Timestamp(end_d + timedelta(days=1), tz="UTC")
    return df[(df.index >= lo) & (df.index < hi)]


def agg_trades(symbol: str, day, market: str = "um") -> pd.DataFrame:
    """Aggregated trades for one UTC day.

    ``is_buyer_maker == True`` means the BUYER was the resting maker order,
    i.e. the aggressor (taker) was the SELLER. See quant_lab.orderflow.signed_volume.
    """
    d = _as_date(day)
    prefix = _market_prefix(market)
    rel = f"{prefix}/daily/aggTrades/{symbol}/{symbol}-aggTrades-{d.isoformat()}.zip"
    cols = AGG_TRADE_COLS_UM if market == "um" else AGG_TRADE_COLS_SPOT
    df = _read_zip_csv(_fetch(rel), cols)
    df["time"] = _to_utc(df["transact_time"])
    df["price"] = df["price"].astype(float)
    df["quantity"] = df["quantity"].astype(float)
    df["is_buyer_maker"] = df["is_buyer_maker"].astype(str).str.lower().eq("true")
    return df.set_index("time").sort_index()


def futures_metrics(symbol: str, day) -> pd.DataFrame:
    """5-minute open interest / long-short ratios for one UTC day (USDT-M futures only)."""
    d = _as_date(day)
    rel = f"futures/um/daily/metrics/{symbol}/{symbol}-metrics-{d.isoformat()}.zip"
    df = _read_zip_csv(_fetch(rel), METRICS_COLS)
    df["create_time"] = pd.to_datetime(df["create_time"], utc=True)
    for c in METRICS_COLS[2:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.set_index("create_time").sort_index()

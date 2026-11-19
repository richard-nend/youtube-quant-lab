from __future__ import annotations

import time


def now_ms() -> int:
    return int(time.time() * 1000)


def f(x) -> float | None:
    """문자열·숫자를 float로. 없거나 변환 불가면 None."""
    if x is None or x == "":
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


class Throttle:
    """키별로 최소 간격(ms)을 지켜 통과시킨다 (호가 스냅샷 저장 간격용)."""

    def __init__(self, interval_ms: int):
        self.interval_ms = interval_ms
        self._last: dict[str, int] = {}

    def allow(self, key: str, ts_ms: int) -> bool:
        last = self._last.get(key)
        if last is not None and ts_ms - last < self.interval_ms:
            return False
        self._last[key] = ts_ms
        return True


# --- instrument_ids.py 에서 필요한 것만 ---
PERP = "PERP"
ASSET_ALIASES = {"UBTC": "BTC", "UETH": "ETH", "USOL": "SOL"}
_QUOTES = ("USDT", "USDC", "FDUSD", "BUSD")


def make_id(base: str, quote: str, market_type: str) -> str:
    return f"{base}-{quote}-{market_type}"


def asset_of(base: str) -> str:
    return ASSET_ALIASES.get(base, base)


def split_concat(symbol: str) -> tuple[str, str]:
    """'BTCUSDT' → ('BTC', 'USDT')."""
    for q in _QUOTES:
        if symbol.endswith(q) and len(symbol) > len(q):
            return symbol[: -len(q)], q
    raise ValueError(f"unknown quote in symbol {symbol!r}")


def binance_usdm(symbol: str) -> str:
    base, quote = split_concat(symbol)
    return make_id(base, quote, PERP)


def toobit_swap(symbol: str) -> str:
    """'BTC-SWAP-USDT' → 'BTC-USDT-PERP'."""
    base, kind, quote = symbol.split("-")
    if kind != "SWAP":
        raise ValueError(f"not a Toobit swap symbol {symbol!r}")
    return make_id(base, quote, PERP)

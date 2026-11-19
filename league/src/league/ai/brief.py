"""AI 팀에 주는 시장 요약. 같은 기록이면 모든 팀에 같은 내용."""
from __future__ import annotations

import json
from datetime import datetime, timezone

import httpx

from ..collect.binance import VENUE as BINANCE
from ..collect.toobit import REST
from ..venues import CFG, TOOBIT

SYMBOLS = {"BTC": ("BTC-SWAP-USDT", "BTCUSDT"), "ETH": ("ETH-SWAP-USDT", "ETHUSDT")}
CTX_WINDOW_MS, BOOK_WINDOW_MS, LIQ_WINDOW_MS = 3_600_000, 600_000, 3_600_000
RULES = (f"BTC·ETH 무기한 선물만 거래한다. BTC+ETH 명목 노출 합계가 순자산의 3배(300%)를 넘을 수 없다. 시장가 수수료 {float(CFG['taker']) * 100:.2f}%, "
         f"지정가 {float(CFG['maker']) * 100:.2f}%. 4시간마다 한 번 판단하고, 판단 사이에는 손절·익절 주문만 작동한다. "
         "손절/익절 가격은 마크가격 기준이다.")


def fetch_candles(symbol: str, interval: str, n: int) -> list[list[float]]:
    """Toobit REST 캔들 → [[시작ms, 시, 고, 저, 종, 거래량], ...] 오래된 것부터. shortcut: 재시도 없음, 실패는 예외로 올림."""
    r = httpx.get(f"{REST}/quote/v1/klines", params={"symbol": symbol, "interval": interval, "limit": n}, timeout=15)
    r.raise_for_status()
    return [[int(k[0])] + [float(x) for x in k[1:6]] for k in r.json()]


def _last(rows: list[dict]) -> dict:
    return rows[-1] if rows else {}


def _imbalance(book: dict) -> float | None:
    bid = sum(s for _, s in book.get("bids", [])[:10])
    ask = sum(s for _, s in book.get("asks", [])[:10])
    return bid / (bid + ask) if bid + ask else None


def _liq(rows: list[dict]) -> dict:
    """바이낸스 강제청산 기록: side=sell 은 롱 청산. 수집기가 아직 이 종류를 만들지 않으면 빈 요약."""
    usd = lambda side: sum(r["price"] * r["size"] for r in rows if r["side"] == side)
    return {"count": len(rows), "long_liq_usd": usd("sell"), "short_liq_usd": usd("buy")}


def build_brief(store, account: dict, history: list[dict], now_ms: int, candles=fetch_candles) -> dict:
    markets = {}
    for name, (tb, bn) in SYMBOLS.items():
        rd = lambda venue, kind, sym, win: [r for r in store.read(venue, kind, now_ms - win, now_ms + 1) if r["symbol"] == sym]
        ctx = _last([r for r in rd(TOOBIT, "ctx", tb, CTX_WINDOW_MS)])
        markets[name] = {
            "candles_4h": candles(tb, "4h", 60), "candles_1h": candles(tb, "1h", 48),
            "mark_price": ctx.get("mark_price"), "funding_rate": ctx.get("funding_rate"), "open_interest": ctx.get("open_interest"),
            "book_imbalance": _imbalance(_last(rd(TOOBIT, "book", tb, BOOK_WINDOW_MS))),
            "liquidations_1h": _liq(rd(BINANCE, "liquidations", bn, LIQ_WINDOW_MS)),
        }
    return {"now_utc": datetime.fromtimestamp(now_ms / 1000, timezone.utc).strftime("%Y-%m-%dT%H:%MZ"),
            "markets": markets, "account": account, "history": history, "rules": RULES}


def brief_json(brief: dict) -> str:
    return json.dumps(brief, ensure_ascii=False, separators=(",", ":"))

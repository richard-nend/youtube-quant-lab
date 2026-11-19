"""AI 팀에 주는 시장 요약. 같은 기록이면 모든 팀에 같은 내용."""
from __future__ import annotations

import json
from datetime import datetime, timezone

import httpx

from ..collect.toobit import REST
from ..venues import CFG, TOOBIT

SYMBOLS = {"BTC": ("BTC-SWAP-USDT", "BTCUSDT"), "ETH": ("ETH-SWAP-USDT", "ETHUSDT")}
CTX_WINDOW_MS, BOOK_WINDOW_MS = 3_600_000, 600_000
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


def build_markets(store, now_ms: int, candles=None) -> dict:
    """시장 부분(모든 팀에 같은 것). 회차마다 한 번 만들어 세 팀에 그대로 준다(리뷰 I5)."""
    candles = candles or fetch_candles   # 호출 때 찾는다(시험에서 바꿔 끼울 수 있게)
    markets = {}
    for name, (tb, bn) in SYMBOLS.items():
        rd = lambda venue, kind, sym, win: [r for r in store.read(venue, kind, now_ms - win, now_ms + 1) if r["symbol"] == sym]
        rows = rd(TOOBIT, "ctx", tb, CTX_WINDOW_MS)
        ctx = _last([r for r in rows if r.get("funding_rate") is not None])   # 1초 마크 행에는 펀딩·OI가 없다
        m = {}
        for iv, n, ms in (("4h", 60, 4 * 3_600_000), ("1h", 48, 3_600_000)):
            c = candles(tb, iv, n)
            m[f"candles_{iv}"] = c
            m[f"candles_{iv}_last_forming"] = bool(c) and c[-1][0] + ms > now_ms   # 마지막 봉이 아직 안 닫혔나
        markets[name] = {
            **m, "mark_price": _last(rows).get("mark_price"), "funding_rate": ctx.get("funding_rate"),
            "open_interest": ctx.get("open_interest"),
            "book_imbalance": _imbalance(_last(rd(TOOBIT, "book", tb, BOOK_WINDOW_MS))),
            "liquidations_1h": None,   # 바이낸스 강제청산 수집기가 없다: 0으로 채우면 "청산 없음"으로 읽힌다
        }
    return markets


def build_brief(store, account: dict, history: list[dict], now_ms: int, candles=None, markets=None) -> dict:
    return {"now_utc": datetime.fromtimestamp(now_ms / 1000, timezone.utc).strftime("%Y-%m-%dT%H:%MZ"),
            "markets": markets if markets is not None else build_markets(store, now_ms, candles),
            "account": account, "history": history, "rules": RULES}


def brief_json(brief: dict) -> str:
    return json.dumps(brief, ensure_ascii=False, separators=(",", ":"))

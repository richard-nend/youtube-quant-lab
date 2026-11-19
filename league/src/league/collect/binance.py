"""바이낸스 USDⓈ-M 수집기 (호가·체결만). AITrading venues/binance_usdm.py 의 파서·웹소켓 부분 복사.

결합 스트림: aggTrade 는 /market, depth20@100ms 는 /public (2026-10-25 서버 실측) → 연결 둘.
"""
from __future__ import annotations

from . import util as ids
from .base import CollectorSpec, VenueCollector
from .models import BookSnapshot, Record, Trade
from .util import Throttle, now_ms

WS_MARKET = "wss://fstream.binance.com/market/stream?streams="
WS_PUBLIC = "wss://fstream.binance.com/public/stream?streams="
VENUE = "binance_usdm"


def _ident(symbol: str) -> dict:
    return {"instrument_id": ids.binance_usdm(symbol), "market_type": ids.PERP}


def parse_agg_trade(d: dict, recv: int) -> Trade:
    return Trade(ts_ms=int(d["T"]), venue=VENUE, symbol=d["s"], price=float(d["p"]),
                 size=float(d["q"]), side="sell" if d.get("m") else "buy",
                 trade_id=str(d["a"]), recv_ms=recv, **_ident(d["s"]))


def parse_depth(d: dict, recv: int, depth: int) -> BookSnapshot:
    return BookSnapshot(ts_ms=int(d.get("T") or d.get("E")), venue=VENUE, symbol=d["s"],
                        bids=[[float(p), float(q)] for p, q in d["b"][:depth]],
                        asks=[[float(p), float(q)] for p, q in d["a"][:depth]], recv_ms=recv,
                        **_ident(d["s"]))


class BinanceUsdmCollector(VenueCollector):
    name = VENUE

    def __init__(self, cfg: CollectorSpec, emit):
        super().__init__(cfg, emit)
        self.depth = cfg.book_depth
        self.book_throttle = Throttle(cfg.book_interval_ms)
        self.syms = {s.upper() for s in cfg.symbols}

    def ws_urls(self) -> list[str]:
        syms = [s.lower() for s in sorted(self.syms)]
        return [WS_MARKET + "/".join(f"{s}@aggTrade" for s in syms),
                WS_PUBLIC + "/".join(f"{s}@depth20@100ms" for s in syms)]

    def handle(self, msg: dict) -> list[Record]:
        d, recv = msg.get("data", msg), now_ms()
        e = d.get("e")
        if e == "aggTrade":
            return [parse_agg_trade(d, recv)]
        if e == "depthUpdate":
            snap = parse_depth(d, recv, self.depth)
            return [snap] if self.book_throttle.allow(snap.symbol, snap.ts_ms) else []
        return []

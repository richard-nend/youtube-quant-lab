"""수집 기록 → 체결 엔진 이벤트. Toobit만 변환(바이낸스는 참고 시세라 None)."""
from __future__ import annotations

from .collect.models import BookSnapshot, Funding, MarketCtx, Record, Trade
from .fill import BookEvent, CtxEvent, FundingEvent, TradeEvent
from .venues import BTC, ETH, TOOBIT

# Toobit 수집기는 심볼을 "BTC-SWAP-USDT" 형태로 기록한다. 바이낸스 형태 키는 무해한 별칭.
INSTRUMENT_OF = {"BTC-SWAP-USDT": BTC, "ETH-SWAP-USDT": ETH, "BTCUSDT": BTC, "ETHUSDT": ETH}


def to_engine_event(rec: Record) -> BookEvent | TradeEvent | CtxEvent | FundingEvent | None:
    iid = INSTRUMENT_OF.get(getattr(rec, "symbol", None))
    if getattr(rec, "venue", None) != TOOBIT or iid is None:
        return None
    if isinstance(rec, BookSnapshot):
        return BookEvent(rec.ts_ms, iid, rec.bids, rec.asks)
    if isinstance(rec, Trade):
        return TradeEvent(rec.ts_ms, iid, rec.price, rec.size, rec.side)
    if isinstance(rec, MarketCtx):
        return CtxEvent(rec.ts_ms, iid, mark=rec.mark_price, index=rec.index_price,
                        funding_rate=rec.funding_rate, next_funding_ms=rec.next_funding_ms,
                        funding_interval_h=rec.funding_interval_h)
    if isinstance(rec, Funding):
        return FundingEvent(rec.settle_ms, iid, rec.rate)
    return None

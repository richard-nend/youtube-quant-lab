"""상품별 시장 상태: 최신 호가(자기 충격 포함)·체결·마크·캔들, 체결 정확도 등급 판정(명세 3절)."""
from __future__ import annotations

from decimal import Decimal

from .events import BookEvent, CandleEvent, CtxEvent, TradeEvent
from .util import ZERO

GRADE_A_MAX_INTERVAL_MS = 2000   # 명세 3절: 스냅샷 간격 ≤ 2초 + 체결 데이터 → A
STALE_BOOK_MS = 60_000           # 호가가 이보다 오래됐고 캔들이 있으면 C로 내린다(수집 공백)


class MarketState:
    def __init__(self, instrument_id: str):
        self.instrument_id = instrument_id
        self.book: BookEvent | None = None
        self.prev_book_ts: int | None = None
        # 스냅샷 안에서 우리 주문이 먹은 양: (쪽, 단계 번호) → 수량. 새 스냅샷이 오면 초기화(명세 5.2 자기 충격)
        self.consumed: dict[tuple[str, int], Decimal] = {}
        self.trades_seen = False
        self.last_price: Decimal | None = None
        self.last_trade_ts: int | None = None
        self.mark: Decimal | None = None
        self.index: Decimal | None = None
        self.funding_rate: Decimal | None = None
        self.next_funding_ms: int | None = None
        self.funding_interval_h: Decimal | None = None
        self.ctx_ts: int | None = None
        self.candle: CandleEvent | None = None
        self.event = None  # 지금 처리 중인 이벤트 (체결 모델이 본다)

    def apply(self, ev) -> None:
        if isinstance(ev, BookEvent):
            self.prev_book_ts = self.book.ts_ms if self.book else None
            self.book = ev
            self.consumed = {}
        elif isinstance(ev, TradeEvent):
            self.trades_seen = True
            self.last_price, self.last_trade_ts = ev.price, ev.ts_ms
        elif isinstance(ev, CtxEvent):
            for k in ("mark", "index", "funding_rate", "next_funding_ms", "funding_interval_h"):
                v = getattr(ev, k)
                if v is not None:
                    setattr(self, k, v)
            self.ctx_ts = ev.ts_ms
        elif isinstance(ev, CandleEvent):
            self.candle = ev
            if not self.trades_seen:
                self.last_price = ev.close

    def book_interval(self) -> int | None:
        if self.book is None or self.prev_book_ts is None:
            return None
        return self.book.ts_ms - self.prev_book_ts

    def grade(self, now_ms: int) -> str | None:
        """A: 호가 간격·나이 ≤ 2초 + 체결 데이터. B: 호가만 또는 간격 > 2초. C: 캔들만(또는 호가 끊김)."""
        if self.book is None:
            return "C" if self.candle else None
        age = now_ms - self.book.ts_ms
        if age > STALE_BOOK_MS and self.candle and self.candle.ts_ms >= self.book.ts_ms:
            return "C"
        iv = self.book_interval()
        if (self.trades_seen and iv is not None and iv <= GRADE_A_MAX_INTERVAL_MS
                and age <= GRADE_A_MAX_INTERVAL_MS):
            return "A"
        return "B"

    def levels(self, taker_side: str) -> tuple:
        """테이커가 먹는 쪽: 매수 → asks, 매도 → bids."""
        if self.book is None:
            return ()
        return self.book.asks if taker_side == "buy" else self.book.bids

    def best_opposite(self, side: str) -> Decimal | None:
        lv = self.levels(side)
        return lv[0][0] if lv else None

    def visible_at(self, side: str, price: Decimal) -> Decimal:
        """같은 쪽(매수면 bids) 해당 가격 단계의 보이는 잔량 — 대기열 위치."""
        if self.book is None:
            return ZERO
        own = self.book.bids if side == "buy" else self.book.asks
        return next((q for p, q in own if p == price), ZERO)

    def ref_price(self) -> Decimal | None:
        """평가용 가격: 마크 → 최종 체결 → 호가 중간 → 캔들 종가."""
        if self.mark is not None:
            return self.mark
        if self.last_price is not None:
            return self.last_price
        if self.book and self.book.bids and self.book.asks:
            return (self.book.bids[0][0] + self.book.asks[0][0]) / 2
        return self.candle.close if self.candle else None

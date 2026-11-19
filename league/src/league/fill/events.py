"""입력 이벤트 (인터페이스 v1 7절). 수집기 Parquet 행(3절)에서 만든다.

정렬 키 = (at_ms, 종류 우선순위, 입력 순번). at_ms는 "이 정보를 쓸 수 있게 되는 시각":
캔들은 봉이 끝나야 알 수 있으므로 at_ms = 봉 시작 + 길이 (미래 정보 금지).
같은 밀리초에서는 마크·펀딩을 먼저 반영하고(청산·트리거 판정), 호가 → 체결 → 캔들 → 시계 → 주문 순.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import ClassVar

from .util import D

PRIORITY = {"ctx": 0, "funding": 1, "book": 2, "trade": 3, "candle": 4, "delist": 5, "clock": 6, "command": 7}


def _levels(xs) -> tuple[tuple[Decimal, Decimal], ...]:
    return tuple((D(p), D(q)) for p, q in xs)


@dataclass(slots=True)
class BookEvent:
    """호가 스냅샷. bids 내림차순, asks 오름차순 [(가격, 수량)], 수량은 코인 단위."""
    ts_ms: int
    instrument_id: str
    bids: tuple
    asks: tuple
    kind: ClassVar[str] = "book"

    def __post_init__(self):
        self.bids, self.asks = _levels(self.bids), _levels(self.asks)

    @property
    def at_ms(self) -> int:
        return self.ts_ms


@dataclass(slots=True)
class TradeEvent:
    """시장 체결. side = 공격 방향(buy = 매수자가 테이커)."""
    ts_ms: int
    instrument_id: str
    price: Decimal
    size: Decimal
    side: str
    kind: ClassVar[str] = "trade"

    def __post_init__(self):
        self.price, self.size = D(self.price), D(self.size)

    @property
    def at_ms(self) -> int:
        return self.ts_ms


@dataclass(slots=True)
class CtxEvent:
    """마크·인덱스(HL 오라클)·예상 펀딩비(정산 1회분)·다음 정산 시각."""
    ts_ms: int
    instrument_id: str
    mark: Decimal | None = None
    index: Decimal | None = None
    funding_rate: Decimal | None = None
    next_funding_ms: int | None = None
    funding_interval_h: Decimal | None = None
    kind: ClassVar[str] = "ctx"

    def __post_init__(self):
        for k in ("mark", "index", "funding_rate", "funding_interval_h"):
            v = getattr(self, k)
            if v is not None:
                setattr(self, k, D(v))

    @property
    def at_ms(self) -> int:
        return self.ts_ms


@dataclass(slots=True)
class FundingEvent:
    """확정 정산 기록(kind=funding). ts_ms = 정산 시각. 있으면 ctx 예상값보다 우선."""
    ts_ms: int
    instrument_id: str
    rate: Decimal
    kind: ClassVar[str] = "funding"

    def __post_init__(self):
        self.rate = D(self.rate)

    @property
    def at_ms(self) -> int:
        return self.ts_ms


@dataclass(slots=True)
class CandleEvent:
    """캔들(등급 C 구간). ts_ms = 봉 시작, 쓸 수 있는 시각 = 봉 끝."""
    ts_ms: int
    instrument_id: str
    interval_ms: int
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal = field(default=Decimal(0))
    kind: ClassVar[str] = "candle"

    def __post_init__(self):
        for k in ("open", "high", "low", "close", "volume"):
            setattr(self, k, D(getattr(self, k)))

    @property
    def at_ms(self) -> int:
        return self.ts_ms + self.interval_ms


@dataclass(slots=True)
class ClockEvent:
    """시장 데이터 없이 시간만 흐르게 할 때 (전략 타이머·펀딩 정산 진행)."""
    ts_ms: int
    instrument_id: str | None = None
    kind: ClassVar[str] = "clock"

    @property
    def at_ms(self) -> int:
        return self.ts_ms


@dataclass(slots=True)
class OrderCommand:
    """외부(전략 실행기·실거래 그림자)에서 들어온 주문·취소. ts_ms = 결정 시각."""
    ts_ms: int
    request: object  # OrderRequest | Cancel
    kind: ClassVar[str] = "command"

    @property
    def instrument_id(self) -> str | None:
        return getattr(self.request, "instrument_id", None)

    @property
    def at_ms(self) -> int:
        return self.ts_ms


@dataclass(slots=True)
class DelistEvent:
    """상장폐지(#62, 엔진 0.3.1): 이 시각에 열린 포지션을 price(마지막 거래 봉 종가 = 정산가)로 닫고, 이후 주문은 거절.
    ts_ms = 마지막 거래 봉의 끝(적재기가 거래량 0 평평한 봉이 시작되는 시각으로 넣는다)."""
    ts_ms: int
    instrument_id: str
    price: Decimal
    kind: ClassVar[str] = "delist"

    def __post_init__(self):
        self.price = D(self.price)

    @property
    def at_ms(self) -> int:
        return self.ts_ms


Event = BookEvent | TradeEvent | CtxEvent | FundingEvent | CandleEvent | DelistEvent | ClockEvent | OrderCommand


def sort_events(events) -> list:
    """재생 순서. 같은 키면 입력 순서(안정 정렬) → 같은 입력이면 같은 순서."""
    return [e for _, _, _, e in sorted(
        ((e.at_ms, PRIORITY[e.kind], i, e) for i, e in enumerate(events)), key=lambda x: x[:3])]

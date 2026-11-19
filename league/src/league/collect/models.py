"""정규화 레코드. 모든 거래소 데이터는 이 형태로 바뀌어 저장된다.

공통 계보 필드: venue(원천), grade(등급), recv_ms(수집 시각).
공통 식별 필드(인터페이스 v1 3절): instrument_id(정규 ID), market_type(PERP/SPOT), schema_v.
수량(size·volume·open_interest)은 모두 기초자산(코인) 단위. 계약 단위 거래소(Toobit)는 수집기에서 환산한다.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

Side = Literal["buy", "sell"]  # 공격(aggressor) 방향

SCHEMA_V = 1


@dataclass(slots=True)
class Trade:
    ts_ms: int
    venue: str
    symbol: str
    price: float
    size: float
    side: Side
    trade_id: str
    recv_ms: int
    grade: str = ""
    instrument_id: str = ""
    market_type: str = "PERP"
    schema_v: int = SCHEMA_V

    kind = "trades"


@dataclass(slots=True)
class BookSnapshot:
    ts_ms: int
    venue: str
    symbol: str
    bids: list[list[float]]  # [[price, size], ...] 높은 가격부터
    asks: list[list[float]]  # [[price, size], ...] 낮은 가격부터
    recv_ms: int
    grade: str = ""
    instrument_id: str = ""
    market_type: str = "PERP"
    schema_v: int = SCHEMA_V

    kind = "book"


@dataclass(slots=True)
class MarketCtx:
    """펀딩·OI·마크·인덱스(오라클) 등 파생 상태."""
    ts_ms: int
    venue: str
    symbol: str
    recv_ms: int
    funding_rate: float | None = None   # 해당 거래소 펀딩 주기(funding_interval_h) 기준 비율
    mark_price: float | None = None
    index_price: float | None = None    # 인덱스 또는 오라클 가격
    open_interest: float | None = None  # 기초자산(코인) 단위
    premium: float | None = None
    next_funding_ms: int | None = None
    grade: str = ""
    instrument_id: str = ""
    market_type: str = "PERP"
    schema_v: int = SCHEMA_V
    funding_interval_h: float | None = None  # HL 1, 바이낸스 8(종목별 변경 가능)

    kind = "ctx"


@dataclass(slots=True)
class Funding:
    """펀딩 정산 이력 (ts_ms = settle_ms)."""
    ts_ms: int
    venue: str
    symbol: str
    rate: float
    interval_h: float
    settle_ms: int
    recv_ms: int
    premium: float | None = None
    grade: str = ""
    instrument_id: str = ""
    market_type: str = "PERP"
    schema_v: int = SCHEMA_V

    kind = "funding"


@dataclass(slots=True)
class Candle:
    """캔들 (ts_ms = 시작 시각). origin: api(거래소 제공) / derived(자체 생성)."""
    ts_ms: int
    venue: str
    symbol: str
    interval: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    recv_ms: int
    quote_volume: float | None = None
    trades_n: int | None = None
    origin: str = "api"
    grade: str = ""
    instrument_id: str = ""
    market_type: str = "PERP"
    schema_v: int = SCHEMA_V
    gap_ms: int | None = None  # derived 전용: 봉 안 수집 공백(ms). api 는 None
    n_sources: int | None = None  # venue=composite 전용: 봉 안에서 지수를 만든 최소 거래소 수
    method_v: str | None = None   # venue=composite 전용: 지수 방법 버전

    kind = "candles"


@dataclass(slots=True)
class Instrument:
    """상품 메타 일별 스냅샷 (인터페이스 v1 2절 `instruments`).

    venue_symbol = symbol, type = market_type (공통 열과 같은 값이라 따로 두지 않음).
    """
    ts_ms: int
    venue: str
    symbol: str
    recv_ms: int
    asset: str
    quote: str
    tick_size: float | None = None
    lot_size: float | None = None
    min_notional: float | None = None
    max_leverage: float | None = None
    margin_tiers: str | None = None        # JSON 문자열 (거래소 원형)
    contract_multiplier: float | None = None
    status: str | None = None
    listed_ms: int | None = None
    delisted_ms: int | None = None
    updated_ms: int | None = None
    grade: str = ""
    instrument_id: str = ""
    market_type: str = "PERP"
    schema_v: int = SCHEMA_V

    kind = "instruments"


Record = Trade | BookSnapshot | MarketCtx | Funding | Candle | Instrument


def to_row(rec: Record) -> dict:
    return asdict(rec)

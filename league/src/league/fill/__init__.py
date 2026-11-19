"""AITrading 공통 체결 엔진 — 백테스트·페이퍼·그림자 체결이 같은 엔진을 쓴다.

명세: docs/specs/fill-engine.md (v0.1), 경계: docs/specs/interfaces-v1.md 7절.
돈·수량은 Decimal(인터페이스 v1 1절). 시세 입력(float)은 이벤트를 만들 때 Decimal로 바꾼다.
"""
ENGINE_VERSION = "0.3.2"   # 0.3.2 (11-12): 표본 간격 경계(일간 = 자정)마다 자산 표본 — 낮 체결이 있으면 그날 마지막 체결 시각 값이 일간 값이던 것 · 등급 C 지수(-INDEX) 거래량 0은 반 스프레드만 · 0.3.1 (#62): 상장폐지 사건 — 정산가로 닫고 이후 주문 거절 · 0.3.0 (#61): 캔들만 있는 구간 봉 안 극값 청산 · 0.2.1 (11-05 Q53): 자산 곡선 같은 시각 정산 뒤 기록 기본 · 0.2.2 (#56): 청산 뒤 음수 잔고 보험 기금 처리

from .engine import Cancel, Engine, EngineConfig, EngineView, RunResult, ScriptedStrategy  # noqa: E402
from .events import (BookEvent, CandleEvent, ClockEvent, CtxEvent, DelistEvent, FundingEvent,  # noqa: E402
                     OrderCommand, TradeEvent)
from .instruments import FeeSchedule, InstrumentSpec  # noqa: E402
from .orders import OrderRequest, Trigger  # noqa: E402

__all__ = [
    "ENGINE_VERSION", "Engine", "EngineConfig", "EngineView", "RunResult", "ScriptedStrategy", "Cancel",
    "BookEvent", "TradeEvent", "CtxEvent", "FundingEvent", "CandleEvent", "DelistEvent", "ClockEvent", "OrderCommand",
    "FeeSchedule", "InstrumentSpec", "OrderRequest", "Trigger",
]

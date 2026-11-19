"""청산가 공식 (명세 8절). 엔진의 청산 판정은 계좌 자산 < 유지 증거금으로 하고,
이 공식들은 order_preview(5주차)·시험에서 같은 정의를 쓰기 위한 순수 함수다."""
from __future__ import annotations

from decimal import Decimal

from .util import ONE


def hl_liquidation_price(price: Decimal, size: Decimal, side: int, margin_available: Decimal,
                         maint_rate: Decimal) -> Decimal:
    """HL: liq = price − side × margin_available / size / (1 − l × side), l = 유지 증거금률.

    margin_available — 교차: 계좌 가치 − 유지 증거금, 격리: 격리 증거금 − 유지 증거금 (진입 시점 기준).
    side 롱 1 · 숏 −1, size > 0.
    """
    return price - side * margin_available / size / (ONE - maint_rate * side)


def binance_liquidation_price(wallet: Decimal, pos: Decimal, entry: Decimal, side: int, mmr: Decimal,
                              cum: Decimal = Decimal(0), other_maint: Decimal = Decimal(0),
                              other_upnl: Decimal = Decimal(0)) -> Decimal:
    """바이낸스 한 방향 모드: LP = (WB − TMM1 + UPNL1 + cumB − Side×Pos×EP) / (Pos×MMR − Side×Pos).

    격리: wallet = 격리 잔고, other_* = 0. 교차: wallet = crossWalletBalance, other_* = 다른 포지션.
    cum = 티어 유지 금액. pos > 0.
    """
    return (wallet - other_maint + other_upnl + cum - side * pos * entry) / (pos * mmr - side * pos)

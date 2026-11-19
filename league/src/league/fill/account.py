"""계좌: 현금(견적 자산) + 상품별 포지션(한 방향 모드). 교차 증거금 하나 (격리는 v1 이후)."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from .util import ZERO, sign


@dataclass
class Position:
    qty: Decimal = ZERO     # 롱 +, 숏 −
    entry: Decimal = ZERO   # 평균 진입가


class Account:
    def __init__(self, cash: Decimal):
        self.cash = cash
        self.pos: dict[str, Position] = {}

    def qty(self, iid: str) -> Decimal:
        p = self.pos.get(iid)
        return p.qty if p else ZERO

    def apply_fill(self, iid: str, side: str, qty: Decimal, price: Decimal) -> Decimal:
        """체결 반영 → 실현 손익(현금에 더함). 반대 방향이 포지션보다 크면 뒤집고 남은 양으로 새 진입."""
        p = self.pos.setdefault(iid, Position())
        d = qty if side == "buy" else -qty
        realized = ZERO
        if p.qty == 0 or sign(p.qty) == sign(d):
            total = abs(p.qty) + qty
            p.entry = (p.entry * abs(p.qty) + price * qty) / total
            p.qty += d
        else:
            close = min(qty, abs(p.qty))
            realized = (price - p.entry) * close * sign(p.qty)
            p.qty += d
            if p.qty == 0:
                p.entry = ZERO
            elif sign(p.qty) == sign(d):  # 뒤집힘
                p.entry = price
        self.cash += realized
        return realized

    def upnl(self, iid: str, mark: Decimal) -> Decimal:
        p = self.pos.get(iid)
        return (mark - p.entry) * p.qty if p and p.qty else ZERO

    def open_positions(self) -> list[tuple[str, Position]]:
        return [(k, v) for k, v in sorted(self.pos.items()) if v.qty != 0]

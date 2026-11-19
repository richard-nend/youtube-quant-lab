"""우리 팀 크기: 손절까지 위험 = 순자산 1%, 열린 명목 합계가 순자산 3배를 넘으면 축소."""
from decimal import ROUND_DOWN, Decimal

from ..decision import STEP  # lot step: venues에 규격이 없어 decision과 같은 값을 쓴다

RISK = Decimal("0.01")
CAP = Decimal(3)


def size_qty(equity: Decimal, entry: Decimal, stop: Decimal, open_notional: Decimal) -> Decimal:
    dist = abs(entry - stop)
    if dist == 0 or entry <= 0:
        return Decimal(0)
    room = max(Decimal(0), CAP * equity - open_notional)
    return min(equity * RISK / dist, room / entry).quantize(STEP, ROUND_DOWN)

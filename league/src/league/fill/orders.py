"""주문 요청·주문 상태 (인터페이스 v1 5절, schemas/trading/order.json).

상태 전이는 apply_transition 하나로만 한다. 시뮬레이션의 흐름:
created → submitted(결정 시각) → acknowledged(도착 시각) → partially_filled → filled | canceled | rejected | expired.
unknown은 실거래 전송 전용(응답 없음) — 엔진은 만들지 않지만 전이 표는 공통으로 둔다.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal

from .util import ZERO, D, dstr

ORDER_TYPES = ("market", "limit", "stop_market", "stop_limit", "take_profit_market", "take_profit_limit")
TIFS = ("GTC", "IOC", "ALO")
FINAL = frozenset({"filled", "canceled", "rejected", "expired"})

TRANSITIONS: dict[str, frozenset[str]] = {
    "created": frozenset({"submitted", "rejected"}),
    "submitted": frozenset({"acknowledged", "rejected", "canceled", "unknown"}),
    "acknowledged": frozenset({"partially_filled", "filled", "canceled", "rejected", "expired"}),
    "partially_filled": frozenset({"partially_filled", "filled", "canceled", "expired"}),
    "unknown": frozenset({"acknowledged", "partially_filled", "filled", "canceled", "rejected", "expired"}),
}


class TransitionError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class Trigger:
    price: Decimal
    by: str | None = None  # "mark" | "last". None → 거래소 기본값(설정)

    def __post_init__(self):
        object.__setattr__(self, "price", D(self.price))
        if self.by not in (None, "mark", "last"):
            raise ValueError(f"trigger.by {self.by!r}")


@dataclass(frozen=True, slots=True)
class OrderRequest:
    instrument_id: str
    side: str                     # buy | sell
    type: str                     # ORDER_TYPES
    qty: Decimal
    price: Decimal | None = None  # 지정가(limit 계열)
    tif: str | None = None        # limit 계열만. 기본 GTC
    trigger: Trigger | None = None
    reduce_only: bool = False
    builder_fee_bps: Decimal | None = None  # HL 빌더 코드 비율(bp)
    tag: str | None = None        # 전략 쪽 이름(취소할 때 씀)

    def __post_init__(self):
        object.__setattr__(self, "qty", D(self.qty))
        if self.price is not None:
            object.__setattr__(self, "price", D(self.price))
        if self.builder_fee_bps is not None:
            object.__setattr__(self, "builder_fee_bps", D(self.builder_fee_bps))
        if self.side not in ("buy", "sell"):
            raise ValueError(f"side {self.side!r}")
        if self.type not in ORDER_TYPES:
            raise ValueError(f"type {self.type!r}")
        if self.qty <= 0:
            raise ValueError("qty must be > 0")
        limit_like = self.type.endswith("limit")
        if limit_like and self.price is None:
            raise ValueError(f"{self.type} needs price")
        if self.tif is not None and (not limit_like or self.tif not in TIFS):
            raise ValueError(f"tif {self.tif!r} for {self.type}")
        if self.type.startswith(("stop", "take_profit")) and self.trigger is None:
            raise ValueError(f"{self.type} needs trigger")


@dataclass
class Order:
    """엔진 안의 주문. 체결 모델이 읽고(fillable·queue_ahead) 엔진이 상태를 바꾼다."""
    ord_id: str
    req: OrderRequest
    created_ms: int
    arrival_ms: int
    state: str = "created"
    updated_ms: int = 0
    filled: Decimal = ZERO
    arrived: bool = False          # 체결 모델이 도착 처리(테이커 부분·대기열 위치)를 했는가
    triggered: bool = False        # 스탑·익절 발동
    trigger_by: str | None = None  # 실제 쓰는 기준(요청값 또는 거래소 기본값)
    liquidation: bool = False      # 거래소 청산 주문(슬리피지 한도 없음, 원장 type=liquidation)
    queue_ahead: Decimal | None = None
    max_fill: Decimal | None = None  # reduce-only 상한(엔진이 매번 설정)
    reason: str | None = None      # 거절·취소 사유
    history: list[tuple[int, str]] = field(default_factory=list)

    @property
    def side(self) -> str:
        return self.req.side

    @property
    def limit(self) -> Decimal | None:
        return self.req.price

    @property
    def is_limit(self) -> bool:
        return self.req.type.endswith("limit")

    @property
    def is_conditional(self) -> bool:
        return self.req.type.startswith(("stop", "take_profit"))

    @property
    def tif(self) -> str:
        return self.req.tif or "GTC"

    @property
    def remaining(self) -> Decimal:
        return self.req.qty - self.filled

    @property
    def fillable(self) -> Decimal:
        r = self.remaining
        return r if self.max_fill is None else min(r, self.max_fill)

    @property
    def live(self) -> bool:
        return self.state not in FINAL

    def to_row(self, run_id: str) -> dict:
        """schemas/trading/order.json 행."""
        r = self.req
        return {
            "ord_id": self.ord_id, "run_id": run_id, "acc_id": None, "instrument_id": r.instrument_id,
            "side": r.side, "type": r.type, "tif": (self.tif if self.is_limit else None),
            "qty": dstr(r.qty, None), "price": dstr(r.price, None) if r.price is not None else None,
            "trigger": ({"price": dstr(r.trigger.price, None), "by": self.trigger_by or r.trigger.by or "mark"}
                        if r.trigger else None),
            "reduce_only": r.reduce_only, "idempotency_key": f"sim-{self.ord_id}", "state": self.state,
            "venue_order_id": None,
            "builder_fee_bps": dstr(r.builder_fee_bps, None) if r.builder_fee_bps is not None else None,
            "created_ms": self.created_ms, "updated_ms": self.updated_ms,
        }


def apply_transition(o: Order, new: str, ts_ms: int, reason: str | None = None) -> None:
    """모든 상태 전이의 단일 입구(인터페이스 v1 5절)."""
    if new not in TRANSITIONS.get(o.state, frozenset()):
        raise TransitionError(f"{o.ord_id}: {o.state} → {new}")
    o.state, o.updated_ms = new, ts_ms
    if reason:
        o.reason = reason
    o.history.append((ts_ms, new))

"""체결 모델 A·B·C — 같은 인터페이스 `fill(order, market_state) -> list[Leg]` (인터페이스 v1 7절).

모델은 지금 처리 중인 이벤트(market_state.event)를 보고 이 주문의 체결 조각을 돌려준다.
도착 처리(order.arrived=False → True)에서 테이커 부분과 대기열 위치를 정하고, 그 뒤엔 대기 체결만 본다.
ALO가 즉시 체결 가능하면 체결 없이 order.reason="alo_marketable"을 남긴다(엔진이 거절 처리).
보수적 기본값(명세 1절): 애매하면 덜 체결되고 더 비싸게.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from .events import BookEvent, CandleEvent, TradeEvent
from .market import MarketState
from .orders import Order
from .util import ONE, ZERO

BP = Decimal("0.0001")


@dataclass(frozen=True, slots=True)
class Leg:
    price: Decimal
    qty: Decimal
    liquidity: str           # maker | taker
    flags: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class FillParams:
    max_slippage: Decimal = Decimal("0.05")     # 시장가 최대 슬리피지 (5.2, HL 프론트 수준 미확인 → 5%)
    depth_penalty: Decimal = Decimal("0.001")   # depth 소진 뒤 가상 단계마다 10bp (5.2)
    c_half_spread_bps: Decimal = Decimal(2)     # 등급 C 반 스프레드 (5.5)
    c_impact_bps_per_pct: Decimal = Decimal(1)  # 등급 C 규모 영향: 봉 거래량 1%당 1bp (어시스턴트 가정, 보정 전)


def _sgn(side: str) -> int:
    return 1 if side == "buy" else -1


def _beyond(price: Decimal, cap: Decimal | None, sgn: int) -> bool:
    return cap is not None and sgn * (price - cap) > 0


def take(st: MarketState, side: str, qty: Decimal, cap: Decimal | None,
         penalty: Decimal) -> tuple[list[tuple[Decimal, Decimal]], Decimal, int]:
    """반대편 호가를 위에서부터 소진. 반환 (조각[(가격, 수량)], 남은 수량, 가상 단계 조각 수).
    가상 단계 조각은 조각 목록의 끝에 온다.

    cap: 이 가격을 넘는 단계는 먹지 않음(지정가·최대 슬리피지). 스냅샷 안 소진량은 st.consumed에 쌓여
    같은 스냅샷의 다음 주문이 같은 잔량을 다시 먹지 못한다(자기 충격).
    depth 20을 다 먹으면 평균 단계 크기의 가상 단계를 마지막 가격에서 단계마다 penalty씩 불리하게 이어 붙인다.
    """
    sgn = _sgn(side)
    key = "a" if side == "buy" else "b"
    levels = st.levels(side)
    legs: list[tuple[Decimal, Decimal]] = []
    rem = qty
    for i, (p, q) in enumerate(levels):
        if rem <= 0 or _beyond(p, cap, sgn):
            return legs, rem, 0
        avail = q - st.consumed.get((key, i), ZERO)
        if avail <= 0:
            continue
        x = min(avail, rem)
        st.consumed[(key, i)] = st.consumed.get((key, i), ZERO) + x
        legs.append((p, x))
        rem -= x
    if rem <= 0 or not levels:
        return legs, rem, 0
    n_real = len(legs)
    avg = sum((q for _, q in levels), ZERO) / len(levels)
    last = levels[-1][0]
    j = 1
    while rem > 0 and avg > 0 and j <= 10_000:
        p = last * (ONE + sgn * penalty * j)
        if _beyond(p, cap, sgn):
            break
        k = (key, len(levels) - 1 + j)
        avail = avg - st.consumed.get(k, ZERO)
        if avail > 0:
            x = min(avail, rem)
            st.consumed[k] = st.consumed.get(k, ZERO) + x
            legs.append((p, x))
            rem -= x
        j += 1
    return legs, rem, len(legs) - n_real


class FillModel:
    grade = "?"

    def __init__(self, params: FillParams):
        self.p = params

    def fill(self, order: Order, st: MarketState) -> list[Leg]:
        raise NotImplementedError


class _BookModel(FillModel):
    """A·B 공통: 도착 시 첫 스냅샷(ts ≥ 도착 시각)에서 호가 소진."""

    def fill(self, o: Order, st: MarketState) -> list[Leg]:
        ev = st.event
        if not o.arrived:
            if not isinstance(ev, BookEvent) or ev.ts_ms < o.arrival_ms:
                return []
            return self._arrive(o, st)
        if o.is_limit:
            return self._rest(o, st)
        return []

    def _arrive(self, o: Order, st: MarketState) -> list[Leg]:
        o.arrived = True
        sgn = _sgn(o.side)
        best = st.best_opposite(o.side)
        if o.is_limit:
            marketable = best is not None and sgn * (best - o.limit) <= 0
            if o.tif == "ALO" and marketable:
                o.reason = "alo_marketable"
                return []
            cap = o.limit
        else:
            if best is None:
                return []
            cap = None if o.liquidation else best * (ONE + sgn * self.p.max_slippage)
        legs, rem, n_virtual = take(st, o.side, o.fillable, cap, self.p.depth_penalty)
        if o.is_limit and rem > 0:
            o.queue_ahead = st.visible_at(o.side, o.limit)
        out = [Leg(p, q, "taker", ("depth_exhausted",) if i >= len(legs) - n_virtual else ())
               for i, (p, q) in enumerate(legs)]
        if not o.is_limit and rem > 0 and out:
            out[-1] = Leg(out[-1].price, out[-1].qty, "taker", out[-1].flags + ("slippage_capped",))
        return out

    def _rest(self, o: Order, st: MarketState) -> list[Leg]:
        raise NotImplementedError


class FillModelA(_BookModel):
    """호가 소진 + 대기열: 반대 방향 체결이 지정가를 관통하면 전부, 같은 가격이면 대기열을 줄인 뒤 초과분."""
    grade = "A"

    def _rest(self, o: Order, st: MarketState) -> list[Leg]:
        t = st.event
        if not isinstance(t, TradeEvent) or t.side == o.side:
            return []
        sgn = _sgn(o.side)
        if sgn * (o.limit - t.price) > 0:  # 매수 지정가 > 체결가 (관통)
            return [Leg(o.limit, o.fillable, "maker")]
        if t.price != o.limit:
            return []
        ahead = (o.queue_ahead or ZERO) - t.size
        if ahead >= 0:
            o.queue_ahead = ahead
            return []
        o.queue_ahead = ZERO
        x = min(-ahead, o.fillable)
        return [Leg(o.limit, x, "maker")] if x > 0 else []


class FillModelB(_BookModel):
    """호가만: 반대편 최우선가가 지정가를 관통한 스냅샷에서만 체결(닿기만 하면 안 됨)."""
    grade = "B"

    def _rest(self, o: Order, st: MarketState) -> list[Leg]:
        ev = st.event
        if not isinstance(ev, BookEvent):
            return []
        best = st.best_opposite(o.side)
        if best is not None and _sgn(o.side) * (o.limit - best) > 0:
            return [Leg(o.limit, o.fillable, "maker")]
        return []


class FillModelC(FillModel):
    """캔들만: 도착 이후 시작한 첫 봉의 시가 ± (반 스프레드 + 규모 영향). 지정가는 고가·저가 관통 시."""
    grade = "C"

    def slippage(self, qty: Decimal, volume: Decimal, instrument_id: str = "") -> Decimal:
        # 지수(현물 다리 대용, 예 BTC-USD-INDEX)는 봉 거래량이 늘 0 — 영향 없이 반 스프레드만(문서화한 고정 스프레드, 11-12 연구 요청)
        pct = (qty / volume * 100) if volume > 0 else Decimal(0) if instrument_id.endswith("-INDEX") else Decimal(100)
        s = (self.p.c_half_spread_bps + self.p.c_impact_bps_per_pct * pct) * BP
        return min(s, self.p.max_slippage)

    def fill(self, o: Order, st: MarketState) -> list[Leg]:
        bar = st.event
        if not isinstance(bar, CandleEvent):
            return []
        sgn = _sgn(o.side)
        out: list[Leg] = []
        if not o.arrived:
            if bar.ts_ms < o.arrival_ms:
                return []
            o.arrived = True
            px = bar.open * (ONE + sgn * self.slippage(o.fillable, bar.volume, o.req.instrument_id))
            if not o.is_limit:
                return [Leg(px, o.fillable, "taker", ("candle_estimate",))]
            if sgn * (bar.open - o.limit) <= 0:  # 시가에서 바로 체결 가능
                if o.tif == "ALO":
                    o.reason = "alo_marketable"
                    return []
                p = min(px, o.limit) if sgn > 0 else max(px, o.limit)
                return [Leg(p, o.fillable, "taker", ("candle_estimate",))]
        if o.is_limit:
            through = bar.low < o.limit if sgn > 0 else bar.high > o.limit
            if through:
                out.append(Leg(o.limit, o.fillable, "maker", ("candle_estimate",)))
        return out


def make_models(params: FillParams) -> dict[str, FillModel]:
    return {"A": FillModelA(params), "B": FillModelB(params), "C": FillModelC(params)}

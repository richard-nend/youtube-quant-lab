"""주문 전 계산 `order_preview` (도구 목록 모듈 5, 빌드 계획 T14). **주문하지 않는다.**

체결 엔진과 같은 정의를 쓴다: 시장가는 같은 호가 소진(`fillmodels.take`, depth 소진 벌점·최대 슬리피지 포함),
수수료 표·유지 증거금률은 `InstrumentSpec`·`DEFAULT_FEES`, 청산 판정은 엔진과 같은 "계좌 자산 < 유지 증거금".

청산가(교차, 어시스턴트 정리): 주문 뒤 계좌 자산 E', 이 상품 수량 q(롱 +, 숏 −), 마크 m, 유지율 l,
다른 포지션 유지 증거금 OM, 유지 금액 c 일 때 E' + q(P − m) = OM + |q|·P·l − c 를 P로 푼다.
P = (OM − c − E' + q·m) / (q − |q|·l). 다른 포지션 가격은 고정으로 본다(가정에 표시). OM = c = 0이면 HL 공식
(margin.hl_liquidation_price)과 같다 — 시험으로 고정.

도달 확률: 추세 없는 로그 정규 가정의 장벽 도달 확률 2·(1 − Φ(|ln(L/P)| / (σ√T))). σ = 입력 연 변동성(실현).
과거 변동성 기준 참고값이지 예측이 아니다(사실 원칙 — 방법을 결과에 적는다).
"""
from __future__ import annotations

import math
from decimal import Decimal

from .events import BookEvent
from .fillmodels import BP, FillParams, take
from .instruments import DEFAULT_FEES, FeeSchedule, InstrumentSpec
from .market import MarketState
from .util import ONE, ZERO, D, dstr

PREVIEW_VERSION = "0.1.0"


def _phi(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def touch_probability(price: Decimal, barrier: Decimal, vol_annual: float, days: float) -> float | None:
    """price에서 시작해 days일 안에 barrier에 닿을 확률(추세 없는 로그 정규, 장벽 반사)."""
    if price <= 0 or barrier <= 0 or vol_annual <= 0 or days <= 0:
        return None
    z = abs(math.log(float(barrier / price))) / (vol_annual * math.sqrt(days / 365))
    return min(1.0, 2 * (1 - _phi(z)))


def order_preview(spec: InstrumentSpec, side: str, qty, *, mark, equity, book: BookEvent | None = None,
                  order_type: str = "market", limit_price=None, position_qty=0, position_entry=None,
                  other_maintenance=0, funding_rate=None, funding_interval_h=None, fees: FeeSchedule | None = None,
                  builder_fee_bps=None, fill: FillParams | None = None, vol_annual: float | None = None,
                  horizon_days: float = 1.0, stop_price=None, loss_limit=None) -> dict:
    """주문 하나의 예상 비용·청산가·도달 확률·최대 손실·손실 한도 기준 규모.

    qty: 코인 수량(> 0). equity: 지금 계좌 자산(미실현 포함). position_qty: 이 상품의 기존 수량(롱 +, 숏 −).
    funding_rate: 정산 1회분 예상 펀딩비(ctx), funding_interval_h: 정산 간격(HL 1, 바이낸스 8).
    """
    if side not in ("buy", "sell"):
        raise ValueError("side must be buy|sell")
    qty, mark, equity = D(qty), D(mark), D(equity)
    if qty <= 0 or mark <= 0:
        raise ValueError("qty and mark must be > 0")
    fill = fill or FillParams()
    fees = fees or DEFAULT_FEES[spec.fee_key]
    sgn = 1 if side == "buy" else -1
    flags: list[str] = []
    assumptions: dict = {"liquidation": "cross margin, other positions' prices held constant",
                         "touch_probability": "driftless lognormal barrier, sigma = input realized vol"}

    # ---- 예상 체결 ----
    mid = None
    if book is not None and book.bids and book.asks:
        mid = (book.bids[0][0] + book.asks[0][0]) / 2
    ref = mid or mark
    liquidity, filled, legs_px = "taker", qty, None
    if order_type == "market":
        if book is None or not (book.asks if side == "buy" else book.bids):
            raise ValueError("market preview needs a book snapshot")
        st = MarketState(spec.instrument_id)
        st.apply(book)
        best = st.best_opposite(side)
        cap = best * (ONE + sgn * fill.max_slippage)
        legs, rem, virtual = take(st, side, qty, cap, fill.depth_penalty)
        filled = qty - rem
        if virtual:
            flags.append("depth_exhausted")
        if rem > 0:
            flags.append("slippage_capped")
        if filled <= 0:
            raise ValueError("nothing fillable within max slippage")
        legs_px = sum((p * q for p, q in legs), ZERO) / filled
        price = legs_px
    elif order_type == "limit":
        if limit_price is None:
            raise ValueError("limit preview needs limit_price")
        price = D(limit_price)
        best = None
        if book is not None:
            lv = book.asks if side == "buy" else book.bids
            best = lv[0][0] if lv else None
        if best is not None and sgn * (price - best) >= 0:
            flags.append("limit_marketable")   # 도착 시 테이커로 체결될 수 있음 → 보수적으로 테이커 수수료
        else:
            liquidity = "maker"
            assumptions["limit_fill"] = "assumes full fill at limit price (not guaranteed)"
    else:
        raise ValueError("order_type must be market|limit")

    notional = price * filled
    fee_rate = fees.taker if liquidity == "taker" else fees.maker
    fee = notional * fee_rate
    builder = notional * D(builder_fee_bps) * BP if builder_fee_bps else ZERO
    slip_bp = sgn * (price - ref) / ref / BP
    slip_cost = sgn * (price - mark) * filled          # 체결 즉시 마크 기준 평가 손익(불리하면 +)

    # ---- 주문 뒤 포지션 ----
    q0 = D(position_qty)
    q1 = q0 + sgn * filled
    e1 = equity - fee - builder - slip_cost
    out_pos = {"qty_before": dstr(q0), "qty_after": dstr(q1)}

    liq = dist = p_touch = None
    lev = None
    funding_day = None
    if not spec.is_spot:
        l = spec.maint_rate()
        maint_after = spec.maintenance(q1, mark) + D(other_maintenance)
        if q1 != 0:
            den = q1 - abs(q1) * l
            p = (D(other_maintenance) - spec.maint_amount - e1 + q1 * mark) / den
            liq = p if p > 0 else None
            if liq is not None:
                dist = abs(liq - mark) / mark
                if vol_annual:
                    p_touch = touch_probability(mark, liq, vol_annual, horizon_days)
            lev = abs(q1) * mark / e1 if e1 > 0 else None
            if spec.max_leverage and lev is not None and lev > spec.max_leverage:
                flags.append("over_max_leverage")
        if e1 <= maint_after:
            flags.append("below_maintenance_after_fill")
        if funding_rate is not None:
            per_day = D(24) / D(funding_interval_h or 1)
            # 양수 펀딩: 롱 지급(+ 비용), 숏 수취. 마크 기준 근사(HL 정산은 오라클 가격)
            funding_day = q1 * mark * D(funding_rate) * per_day
            assumptions["funding"] = "current predicted rate held constant; price = mark (HL settles on oracle)"

    # ---- 최대 손실·규모 ----
    exit_fee_rate = fees.taker + (D(builder_fee_bps) * BP if builder_fee_bps else ZERO)
    max_loss = None
    basis = None
    if stop_price is not None:
        sp = D(stop_price)
        max_loss = filled * max(ZERO, sgn * (price - sp)) + fee + builder + filled * sp * exit_fee_rate
        basis = "stop"
        assumptions["stop"] = "stop fills exactly at stop price (gaps can be worse)"
    elif liq is not None:
        max_loss = filled * abs(price - liq) + fee + builder
        basis = "liquidation (cross: can extend to account equity)"
    sized = None
    if loss_limit is not None and stop_price is not None:
        sp = D(stop_price)
        per_unit = abs(price - sp) + price * (fee_rate + (D(builder_fee_bps) * BP if builder_fee_bps else ZERO)) \
            + sp * exit_fee_rate
        if per_unit > 0:
            sized = D(loss_limit) / per_unit

    return {
        "preview_version": PREVIEW_VERSION,
        "order_placed": False,
        "instrument_id": spec.instrument_id, "side": side, "order_type": order_type,
        "qty": dstr(qty), "fillable_qty": dstr(filled),
        "est_price": dstr(price), "reference_price": dstr(ref), "mark": dstr(mark),
        "slippage_bps": float(round(slip_bp, 3)),
        "cost": {"liquidity": liquidity, "fee": dstr(fee), "builder_fee": dstr(builder),
                 "slippage_vs_mark": dstr(slip_cost), "total": dstr(fee + builder + slip_cost),
                 "funding_per_day": dstr(funding_day) if funding_day is not None else None},
        "position": out_pos,
        "leverage_after": float(round(lev, 3)) if lev is not None else None,
        "liquidation_price": dstr(liq) if liq is not None else None,
        "liquidation_distance_pct": float(round(dist * 100, 3)) if dist is not None else None,
        "liquidation_touch_probability": (round(p_touch, 4) if p_touch is not None else None),
        "horizon_days": horizon_days if p_touch is not None else None,
        "max_loss": dstr(max_loss) if max_loss is not None else None, "max_loss_basis": basis,
        "size_for_loss_limit": dstr(sized) if sized is not None else None,
        "flags": flags, "assumptions": assumptions,
        "fees_used": {"taker": dstr(fees.taker, None), "maker": dstr(fees.maker, None)},
    }

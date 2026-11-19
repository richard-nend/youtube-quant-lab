"""엔진 본체: 이벤트 재생 → 주문 도착·취소·트리거 → 체결 모델 → 수수료·펀딩·청산 → 원장·결과 (명세 2~10절).

백테스트는 run(events, strategy), 페이퍼는 같은 Engine에 실시간 이벤트를 step()으로 넣는다(명세 11절).
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Iterable, Protocol

from . import ENGINE_VERSION
from .account import Account
from .events import CandleEvent, ClockEvent, CtxEvent, DelistEvent, FundingEvent, OrderCommand, TradeEvent, sort_events
from .fillmodels import BP, FillParams, Leg, make_models
from .instruments import DEFAULT_FEES, HL, FeeSchedule, InstrumentSpec
from .market import MarketState
from .orders import Order, OrderRequest, apply_transition
from .util import ZERO, D, dstr, make_id

HOUR_MS = 3_600_000


@dataclass(frozen=True, slots=True)
class Cancel:
    """취소 요청: 전략이 붙인 tag 또는 ord_id."""
    tag: str | None = None
    ord_id: str | None = None

    @property
    def instrument_id(self) -> None:
        return None


@dataclass
class EngineConfig:
    instruments: dict[str, InstrumentSpec]
    initial_cash: Decimal = Decimal(10_000)
    latency_ms: int = 250                                   # 5.1
    fees: dict[str, FeeSchedule] = field(default_factory=dict)  # 거래소별 덮어쓰기(사용자 실제 요율)
    fill: FillParams = field(default_factory=FillParams)
    trigger_by: dict[str, str] = field(default_factory=lambda: {HL: "mark", "binance_usdm": "last"})  # 5.4
    funding_stale_ms: int = 120_000                         # 정산 시각에 ctx가 이보다 오래되면 funding_imputed
    hl_partial_liq_notional: Decimal = Decimal(100_000)     # 8절: 초과 포지션은 20%씩, 30초 대기
    hl_partial_fraction: Decimal = Decimal("0.2")
    hl_partial_wait_ms: int = 30_000
    liq_clearance_fee: dict[str, Decimal] = field(default_factory=dict)  # 바이낸스 청산 정리 수수료(미확인 → 설정)
    equity_sample_ms: int = 60_000
    run_id: str | None = None
    sources: list[dict] = field(default_factory=list)       # 메타: [{"venue","grade","internal_only"}]
    # #54·Q53(11-05): True(기본, 엔진 0.2.1~) 면 run() 이 자산 곡선을 같은 시각 이벤트를 모두 처리한 뒤에만 기록한다.
    # False(0.2.0 까지) 는 그 시각 첫 이벤트 뒤에 기록해, 여러 상품 봉이 같은 시각에 끝나면 한 상품만 새 가격인 순간이 일간 값이 됐다
    # (실데이터 분기물 롤: 연 변동성 21.7% → 7.4%, 총수익 그대로). 옛 숫자 재현용으로만 False. step() 직접 호출(페이퍼)은 무관.
    sample_settled: bool = True


class Strategy(Protocol):
    def on_event(self, ev, view: "EngineView") -> Iterable[OrderRequest | Cancel] | None: ...


@dataclass
class RunResult:
    run_id: str
    orders: list[dict]
    fills: list[dict]
    ledger: list[dict]
    metrics: dict
    meta: dict
    equity_curve: list[tuple[int, str]]

    def to_dict(self) -> dict:
        return {"run_id": self.run_id, "orders": self.orders, "fills": self.fills, "ledger": self.ledger,
                "metrics": self.metrics, "meta": self.meta, "equity_curve": self.equity_curve}


class EngineView:
    """전략이 보는 읽기 전용 창."""

    def __init__(self, eng: "Engine"):
        self._e = eng

    @property
    def now(self) -> int:
        return self._e.now

    @property
    def cash(self) -> Decimal:
        return self._e.account.cash

    def position(self, iid: str) -> Decimal:
        return self._e.account.qty(iid)

    def entry(self, iid: str) -> Decimal:
        """그 상품 보유의 평균 진입가(보유 없으면 0). 실행기 두 다리 장부(#129)가 실제 보유 기준으로 잴 때 쓴다."""
        p = self._e.account.pos.get(iid)
        return p.entry if p and p.qty else Decimal(0)

    def last_fill_price(self, iid: str) -> Decimal | None:
        """그 상품의 마지막 체결가(없으면 None). #129 다리 장부가 체결 때 하위 계좌를 그 값으로 다시 나눈다."""
        return getattr(self._e, "last_fill_px", {}).get(iid)

    def equity(self) -> Decimal:
        return self._e.equity()

    def market(self, iid: str) -> MarketState:
        return self._e.market(iid)

    def open_orders(self, iid: str | None = None) -> list[Order]:
        return [o for o in self._e.orders if o.live and (iid is None or o.req.instrument_id == iid)]


class ScriptedStrategy:
    """정해진 시각에 정해진 주문을 낸다(시험·실거래 주문의 그림자 재생). steps = [(결정 시각, 요청)]."""

    def __init__(self, steps: list[tuple[int, OrderRequest | Cancel]]):
        self.steps = sorted(steps, key=lambda s: s[0])
        self.i = 0

    def on_event(self, ev, view: EngineView):
        out = []
        while self.i < len(self.steps) and self.steps[self.i][0] <= view.now:
            out.append(self.steps[self.i][1])
            self.i += 1
        return out


class Engine:
    def __init__(self, config: EngineConfig):
        self.cfg = config
        self.run_id = config.run_id or make_id("run", 0, 0)
        self.models = make_models(config.fill)
        self.account = Account(D(config.initial_cash))
        self.markets: dict[str, MarketState] = {}
        self.orders: list[Order] = []
        self.by_tag: dict[str, Order] = {}
        self.cancel_at: dict[str, int] = {}
        self.fills: list[dict] = []
        self.last_fill_px: dict[str, Decimal] = {}
        self.ledger: list[dict] = []
        self.seq = 0
        self.now = 0
        self.funding_due: dict[str, int] = {}
        self.funding_records: dict[tuple[str, int], Decimal] = {}
        self.next_liq_ms = 0
        self.delisted: set[str] = set()     # #62 상장폐지된 상품(주문 거절)
        self.flags: Counter = Counter()
        self.grade_count: Counter = Counter()
        self.grade_notional: dict[str, Decimal] = defaultdict(Decimal)   # Decimal() = 0, 저장(pickle) 가능 — 페이퍼 상태
        self.event_counts: Counter = Counter()
        self.first_ms: int | None = None
        self.equity_curve: list[tuple[int, str]] = []
        self._last_sample: int | None = None
        self._ledger_len = 0

    # ---------- 조회 ----------
    def market(self, iid: str) -> MarketState:
        st = self.markets.get(iid)
        if st is None:
            st = self.markets[iid] = MarketState(iid)
        return st

    def __setstate__(self, state: dict) -> None:
        """저장된 옛 판 엔진(페이퍼 pickle)을 되살릴 때 뒤에 생긴 속성을 기본값으로(#62 delisted)."""
        self.__dict__.update(state)
        try:   # 같은 설정으로 새로 만든 엔진의 속성을 빠진 것만 채운다
            for k, v in Engine(self.cfg).__dict__.items():
                self.__dict__.setdefault(k, v)
        except Exception:
            self.__dict__.setdefault("delisted", set())

    def spec(self, iid: str) -> InstrumentSpec:
        try:
            return self.cfg.instruments[iid]
        except KeyError:
            raise ValueError(f"instrument {iid} not in EngineConfig.instruments") from None

    def fee_schedule(self, venue: str) -> FeeSchedule:
        """venue = 거래소 또는 "거래소:SPOT"(InstrumentSpec.fee_key)."""
        fs = self.cfg.fees.get(venue) or DEFAULT_FEES.get(venue)
        if fs is None:
            raise ValueError(f"no fee schedule for {venue} (EngineConfig.fees)")
        return fs

    def mark_of(self, iid: str) -> Decimal | None:
        return self.market(iid).ref_price()

    def equity(self) -> Decimal:
        eq = self.account.cash
        for iid, _ in self.account.open_positions():
            m = self.mark_of(iid)
            if m is not None:
                eq += self.account.upnl(iid, m)
        return eq

    def maintenance(self) -> Decimal:
        mm = ZERO
        for iid, p in self.account.open_positions():
            m = self.mark_of(iid)
            if m is not None:
                mm += self.spec(iid).maintenance(p.qty, m)
        return mm

    # ---------- 실행 ----------
    def run(self, events, strategy: Strategy | None = None) -> RunResult:
        # replay() 가 있으면 이미 정렬된 흐름(백테스트 Events — 봉 객체를 한꺼번에 만들지 않음, 4f)
        evs = events.replay() if hasattr(events, "replay") else iter(sort_events(events))
        ev = next(evs, None)
        while ev is not None:
            nxt = next(evs, None)
            self.step(ev, strategy, sample=nxt is None or nxt.at_ms != ev.at_ms or not self.cfg.sample_settled)
            ev = nxt
        return self.result()

    def step(self, ev, strategy: Strategy | None = None, sample: bool = True) -> None:
        now = ev.at_ms
        if now < self.now:
            raise ValueError(f"event at {now} before engine time {self.now} (sort events first)")
        self.now = now
        if self.first_ms is None:
            self.first_ms = now
        self.event_counts[ev.kind] += 1
        if isinstance(ev, FundingEvent):
            self.funding_records[(ev.instrument_id, ev.ts_ms)] = ev.rate
        self._settle_funding(now)
        self._acknowledge_and_cancel(now)
        if isinstance(ev, OrderCommand):
            self.submit(ev.request, ev.ts_ms)
        elif isinstance(ev, DelistEvent):
            self._delist(ev, now)
        elif not isinstance(ev, ClockEvent) and ev.instrument_id:
            iid = ev.instrument_id
            st = self.market(iid)
            st.apply(ev)
            st.event = ev
            if iid in self.cfg.instruments:
                if self.spec(iid).has_funding:
                    self._schedule_funding(iid, st)
                self._triggers(iid, st, ev)
                self._match(iid, st, now)
                if isinstance(ev, CandleEvent) and st.mark is None:
                    self._check_liquidation_intrabar(ev, now)
                if isinstance(ev, CtxEvent) or (isinstance(ev, CandleEvent) and st.mark is None):
                    self._check_liquidation(now)
        self._cover_negative_balance(now)
        if strategy is not None:
            for req in strategy.on_event(ev, EngineView(self)) or ():
                self.submit(req, now)
        if sample:
            self._sample_equity(now)

    def submit(self, req: OrderRequest | Cancel, ts_ms: int) -> Order | None:
        """주문·취소 접수(결정 시각). 거래소 도착은 결정 + 지연."""
        if isinstance(req, Cancel):
            o = self.by_tag.get(req.tag) if req.tag else next(
                (x for x in self.orders if x.ord_id == req.ord_id), None)
            if o is not None and o.live:
                self.cancel_at.setdefault(o.ord_id, ts_ms + self.cfg.latency_ms)
            return o
        self.seq += 1
        o = Order(make_id("ord", ts_ms, self.seq), req, created_ms=ts_ms,
                  arrival_ms=ts_ms + self.cfg.latency_ms, updated_ms=ts_ms)
        self.orders.append(o)
        if req.tag:
            self.by_tag[req.tag] = o
        if req.instrument_id not in self.cfg.instruments:
            apply_transition(o, "rejected", ts_ms, "unknown_instrument")
            return o
        if req.instrument_id in self.delisted:
            apply_transition(o, "rejected", ts_ms, "delisted")
            self.flags["order_rejected_delisted"] += 1
            return o
        if req.trigger is not None:
            o.trigger_by = req.trigger.by or self.cfg.trigger_by.get(self.spec(req.instrument_id).venue, "mark")
        apply_transition(o, "submitted", ts_ms)
        return o

    def _acknowledge_and_cancel(self, now: int) -> None:
        for o in self.orders:
            if o.state == "submitted" and o.arrival_ms <= now:
                apply_transition(o, "acknowledged", o.arrival_ms)
            c = self.cancel_at.get(o.ord_id)
            if c is not None and c <= now and o.live and o.state != "submitted":
                apply_transition(o, "canceled", c, "user_cancel")

    def _triggers(self, iid: str, st: MarketState, ev) -> None:
        for o in self.orders:
            if (o.req.instrument_id != iid or not o.is_conditional or o.triggered
                    or o.state != "acknowledged"):
                continue
            if isinstance(ev, CandleEvent) and st.book is None:
                hi, lo = ev.high, ev.low          # 등급 C: 봉 안에서 닿았으면 발동
            elif o.trigger_by == "mark" and isinstance(ev, CtxEvent) and ev.mark is not None:
                hi = lo = ev.mark
            elif o.trigger_by == "last" and isinstance(ev, TradeEvent):
                hi = lo = ev.price
            else:
                continue
            trig = o.req.trigger.price
            stop = o.req.type.startswith("stop")
            up = (o.side == "buy") == stop       # 스탑 매수·익절 매도: 가격이 올라 닿으면
            if (up and hi >= trig) or (not up and lo <= trig):
                o.triggered = True
                o.arrived = False
                o.arrival_ms = max(o.arrival_ms, ev.at_ms)
                self.flags[f"trigger_{o.trigger_by}"] += 1

    def _match(self, iid: str, st: MarketState, now: int) -> None:
        grade = st.grade(now)
        if grade is None:
            return
        model = self.models[grade]
        for o in self.orders:
            if o.req.instrument_id != iid or o.state not in ("acknowledged", "partially_filled"):
                continue
            if o.is_conditional and not o.triggered:
                continue
            if o.req.reduce_only:
                pos = self.account.qty(iid)
                allowed = abs(pos) if (pos < 0) == (o.side == "buy") and pos != 0 else ZERO
                if allowed <= 0:
                    apply_transition(o, "canceled", now, "reduce_only_no_position")
                    continue
                o.max_fill = allowed
            was_arrived = o.arrived
            legs = model.fill(o, st)
            if o.reason == "alo_marketable":
                apply_transition(o, "rejected", now)
                continue
            for leg in legs:
                self._apply_leg(o, leg, grade, now)
            if not was_arrived and o.arrived and o.live and (not o.is_limit or o.tif == "IOC"):
                reason = "ioc_remainder" if o.is_limit else (
                    "slippage_capped" if o.filled > 0 else "no_liquidity")
                apply_transition(o, "canceled", now, reason)
            elif o.req.reduce_only and o.live and o.filled > 0 and self.account.qty(iid) == 0:
                apply_transition(o, "canceled", now, "reduce_only_closed")

    def _apply_leg(self, o: Order, leg: Leg, grade: str, now: int) -> None:
        if leg.qty <= 0:
            return
        ltype = "liquidation" if o.liquidation else "fill"
        iid = o.req.instrument_id
        spec = self.spec(iid)
        fs = self.fee_schedule(spec.fee_key)
        notional = leg.price * leg.qty
        fee = notional * (fs.taker if leg.liquidity == "taker" else fs.maker)
        self.seq += 1
        fil_id = make_id("fil", now, self.seq)
        realized = self.account.apply_fill(iid, o.side, leg.qty, leg.price)
        self.account.cash -= fee
        o.filled += leg.qty
        apply_transition(o, "filled" if o.remaining <= 0 else "partially_filled", now)
        self.last_fill_px[iid] = leg.price
        self.fills.append({
            "fil_id": fil_id, "ord_id": o.ord_id, "run_id": self.run_id, "ts_ms": now,
            "price": dstr(leg.price), "qty": dstr(leg.qty), "fee": dstr(fee), "fee_asset": spec.quote,
            "liquidity": leg.liquidity, "fill_grade": grade, "is_shadow": False,
        })
        self._ledger(now, ltype, iid, realized, spec.quote, fil_id)
        self._ledger(now, "fee", iid, -fee, spec.quote, fil_id)
        clearance = D(self.cfg.liq_clearance_fee.get(spec.venue, 0)) if o.liquidation else ZERO
        if clearance:
            cf = notional * clearance
            self.account.cash -= cf
            self._ledger(now, "fee", iid, -cf, spec.quote, f"clearance:{fil_id}")
        if o.req.builder_fee_bps:
            bf = notional * o.req.builder_fee_bps * BP
            self.account.cash -= bf
            self._ledger(now, "builder_fee", iid, -bf, spec.quote, fil_id)
        self.grade_count[grade] += 1
        self.grade_notional[grade] += notional
        for f in leg.flags + (("liquidation_fill",) if o.liquidation else ()):
            self.flags[f] += 1

    def _ledger(self, ts: int, typ: str, iid: str, amount: Decimal, asset: str, ref: str) -> None:
        self.ledger.append({"run_id": self.run_id, "ts_ms": ts, "type": typ, "instrument_id": iid,
                            "amount": dstr(amount), "asset": asset, "ref_id": ref})

    # ---------- 펀딩 (명세 7절) ----------
    def _schedule_funding(self, iid: str, st: MarketState) -> None:
        now = self.now
        due = self.funding_due.get(iid)
        if self.spec(iid).venue == HL:
            if due is None:
                self.funding_due[iid] = (now // HOUR_MS + 1) * HOUR_MS
            return
        nxt = st.next_funding_ms
        if due is None:
            iv = int((st.funding_interval_h or 8) * HOUR_MS)
            self.funding_due[iid] = nxt if nxt and nxt > now else (now // iv + 1) * iv
        elif nxt and now < nxt < due:
            self.funding_due[iid] = nxt   # 정산 주기가 짧아진 종목(4·1시간)

    def _next_due(self, iid: str, due: int) -> int:
        if self.spec(iid).venue == HL:
            return due + HOUR_MS
        st = self.market(iid)
        if st.next_funding_ms and st.next_funding_ms > due:
            return st.next_funding_ms
        return due + int((st.funding_interval_h or 8) * HOUR_MS)

    def _settle_funding(self, now: int) -> None:
        # 정산 시각 T는 T보다 뒤의 첫 이벤트에서 처리 → T 시각의 확정 기록(FundingEvent)과 T 체결이 먼저 반영된다
        for iid in sorted(self.funding_due):
            due = self.funding_due[iid]
            while due < now:
                self._settle_one(iid, due)
                due = self._next_due(iid, due)
            self.funding_due[iid] = due

    def _settle_one(self, iid: str, t: int) -> None:
        q = self.account.qty(iid)
        if q == 0:
            return
        st, spec = self.market(iid), self.spec(iid)
        rate = self.funding_records.get((iid, t))
        if rate is None:
            rate = st.funding_rate
            if rate is None:
                self.flags["funding_missing"] += 1
                return
            if st.ctx_ts is None or t - st.ctx_ts > self.cfg.funding_stale_ms:
                self.flags["funding_imputed"] += 1
        # HL: 오라클(인덱스) 가격, 바이낸스: 마크 가격
        price = st.index if spec.venue == HL else st.mark
        if price is None:
            price = st.ref_price()
            self.flags["funding_price_fallback"] += 1
            if price is None:
                return
        amt = -q * price * rate   # 양수 펀딩비면 롱이 낸다
        self.account.cash += amt
        self._ledger(t, "funding", iid, amt, spec.quote, f"funding:{iid}:{t}")

    # ---------- 청산 (명세 8절) ----------
    def _check_liquidation(self, now: int) -> None:
        """마크 갱신마다: 자산 < 유지 증거금이면 청산 주문을 낸다. 체결은 판정 이후 첫 호가 스냅샷(5.2) —
        판정 시점의 지난 호가로 체결하면 마크보다 유리한 가격에 청산될 수 있다. 진행 중인 청산이 있으면 기다린다."""
        positions = self.account.open_positions()
        if not positions or now < self.next_liq_ms:
            return
        if any(o.liquidation and o.live for o in self.orders):
            return
        eq, mm = self.equity(), self.maintenance()
        if eq >= mm:
            return
        self.flags["liquidation"] += 1
        hl = any(self.spec(i).venue == HL for i, _ in positions)
        if hl and eq < mm * 2 / 3:
            self._backstop(now)
            return
        for iid, p in positions:
            spec = self.spec(iid)
            qty = abs(p.qty)
            if spec.venue == HL and qty * self.mark_of(iid) > self.cfg.hl_partial_liq_notional:
                qty = qty * self.cfg.hl_partial_fraction
                self.next_liq_ms = now + self.cfg.hl_partial_wait_ms
                self.flags["liquidation_partial"] += 1
            self._liq_order(iid, "sell" if p.qty > 0 else "buy", qty, now)

    def _delist(self, ev: DelistEvent, now: int) -> None:
        """상장폐지(#62): 그 종목 대기 주문 취소, 열린 포지션은 정산가로 닫음(거래 수수료 없음, 원장 ref `delisted`)."""
        iid = ev.instrument_id
        self.delisted.add(iid)
        for o in self.orders:
            if o.req.instrument_id == iid and o.live:
                apply_transition(o, "canceled", now, "delisted")
        q = self.account.qty(iid)
        if q == 0 or iid not in self.cfg.instruments:
            return
        spec = self.spec(iid)
        realized = self.account.apply_fill(iid, "sell" if q > 0 else "buy", abs(q), ev.price)
        self._ledger(now, "fill", iid, realized, spec.quote, f"delisted:{iid}")
        self.flags["delisted_settled"] += 1

    def _check_liquidation_intrabar(self, ev: CandleEvent, now: int) -> None:
        """캔들만 있는 구간(#61, 엔진 0.3.0): 봉 안의 불리한 극값(롱 = 저가, 숏 = 고가)에서 자산이 유지 증거금 아래로 내려갔으면
        그 봉 안에서 청산한다. 청산 가격 = 이 종목 가격만 움직일 때 자산 = 유지 증거금이 되는 가격(시가 ~ 극값 사이 이분법),
        시가에서 이미 아래면 시가(갭). 이 종목은 그 가격, 다른 종목은 지금 기준 가격으로 모두 즉시 청산(교차 증거금).
        종전에는 봉 종가로만 판정하고 다음 봉 시가에 체결 → 급등락 하루를 통째로 맞았다."""
        iid = ev.instrument_id
        q = self.account.qty(iid)
        positions = self.account.open_positions()
        if q == 0 or not positions or now < self.next_liq_ms or any(o.liquidation and o.live for o in self.orders):
            return
        spec = self.spec(iid)
        others_eq, others_mm = self.account.cash, ZERO
        for j, p in positions:
            if j == iid:
                continue
            m = self.mark_of(j)
            if m is not None:
                others_eq += self.account.upnl(j, m)
                others_mm += self.spec(j).maintenance(p.qty, m)

        def gap(price: Decimal) -> Decimal:   # 자산 − 유지 증거금, 이 종목 가격만 price 일 때
            return others_eq + self.account.upnl(iid, price) - others_mm - spec.maintenance(q, price)

        worst = D(ev.low) if q > 0 else D(ev.high)
        if gap(worst) >= 0:
            return
        start = D(ev.open)
        if gap(start) < 0:
            price = start
        else:
            good, bad = start, worst
            for _ in range(60):
                mid = (good + bad) / 2
                if gap(mid) >= 0:
                    good = mid
                else:
                    bad = mid
            price = bad          # 보수적으로 극값 쪽 끝
        self.flags["liquidation"] += 1
        self.flags["liquidation_intrabar"] += 1
        grade = self.market(iid).grade(now) or "C"
        for j, p in positions:
            px = price if j == iid else self.mark_of(j)
            if px is None:
                continue
            o = self._liq_order(j, "sell" if p.qty > 0 else "buy", abs(p.qty), now)
            o.arrived = True
            self._apply_leg(o, Leg(px, abs(p.qty), "taker"), grade if j == iid else (self.market(j).grade(now) or "C"), now)

    def _liq_order(self, iid: str, side: str, qty: Decimal, now: int) -> Order:
        """거래소가 내는 청산 시장가 주문(reduce-only). 지연 없이 도착, 테이커 수수료(보수적)."""
        self.seq += 1
        o = Order(make_id("ord", now, self.seq),
                  OrderRequest(iid, side, "market", qty, reduce_only=True, tag=f"liquidation-{self.seq}"),
                  created_ms=now, arrival_ms=now, updated_ms=now, liquidation=True)
        for s in ("submitted", "acknowledged"):
            apply_transition(o, s, now)
        self.orders.append(o)
        return o

    def _cover_negative_balance(self, now: int) -> None:
        """포지션이 모두 닫혔는데 잔고가 음수면 거래소 보험 기금이 메운다(교차 증거금 손실 상한 = 자산). #56 실데이터에서
        바이낸스 청산 주문 체결 뒤 음수가 끝까지 남던 것(HL 백스톱 경로만 0 으로 맞췄음)."""
        if self.account.cash >= 0 or self.account.open_positions():
            return
        self.flags["negative_balance_covered"] += 1
        self._ledger(now, "liquidation", None, -self.account.cash, "USDC", "insurance_fund")
        self.account.cash = ZERO

    def _backstop(self, now: int) -> None:
        """HL 백스톱: 자산이 유지 증거금 2/3 아래 → 마크로 넘기고 남은 증거금은 회수 안 됨(교차는 자산 0)."""
        self.flags["liquidation_backstop"] += 1
        last_spec = None
        for iid, p in self.account.open_positions():
            o = self._liq_order(iid, "sell" if p.qty > 0 else "buy", abs(p.qty), now)
            o.arrived = True
            self._apply_leg(o, Leg(self.mark_of(iid), abs(p.qty), "taker"), self.market(iid).grade(now) or "C", now)
            last_spec = self.spec(iid)
        if self.account.cash != 0 and last_spec is not None:   # 남은 증거금 회수 안 됨, 부족분은 백스톱이 떠안음
            self._ledger(now, "liquidation", last_spec.instrument_id, -self.account.cash, last_spec.quote,
                         "backstop")
            self.account.cash = ZERO

    # ---------- 결과 (명세 9·10절) ----------
    def _sample_equity(self, now: int) -> None:
        changed = len(self.ledger) != self._ledger_len
        step = self.cfg.equity_sample_ms
        # 표본 간격 경계(일간이면 자정)를 넘으면 반드시 기록 — 원장 변경 표본이 시계를 되돌려 자정 값이 빠지고
        # 그날 마지막 체결 시각 자산이 일간 값이 되던 문제(11-12 연구 요청 b)
        if changed or self._last_sample is None or now - self._last_sample >= step or now // step != self._last_sample // step:
            self.equity_curve.append((now, dstr(self.equity())))
            self._last_sample, self._ledger_len = now, len(self.ledger)

    def result(self) -> RunResult:
        sums: dict[str, Decimal] = defaultdict(lambda: ZERO)
        for e in self.ledger:
            sums[e["type"]] += Decimal(e["amount"])
        peak, mdd = None, ZERO
        for _, v in self.equity_curve:
            v = Decimal(v)
            peak = v if peak is None or v > peak else peak
            if peak > 0:
                mdd = max(mdd, (peak - v) / peak)
        n_fill = sum(self.grade_count.values())
        tot_notional = sum(self.grade_notional.values(), ZERO)
        by_count = {g: round(100 * self.grade_count[g] / n_fill, 1) for g in "ABC"} if n_fill else {}
        by_notional = ({g: float(round(100 * self.grade_notional[g] / tot_notional, 1)) for g in "ABC"}
                       if tot_notional else {})
        cfg = self.cfg
        metrics = {
            "initial_cash": dstr(D(cfg.initial_cash)), "final_cash": dstr(self.account.cash),
            "final_equity": dstr(self.equity()),
            "realized_pnl": dstr(sums["fill"] + sums["liquidation"]),
            "fees": dstr(sums["fee"] + sums["builder_fee"]), "funding": dstr(sums["funding"]),
            "max_drawdown_pct": float(round(mdd * 100, 4)),
            "orders": len(self.orders), "fills": len(self.fills),
            "open_positions": [{"instrument_id": i, "qty": dstr(p.qty), "entry": dstr(p.entry)}
                               for i, p in self.account.open_positions()],
        }
        meta = {
            "engine_version": ENGINE_VERSION, "run_id": self.run_id,
            "data": {"start_ms": self.first_ms, "end_ms": self.now, "events": dict(sorted(self.event_counts.items())),
                     "instruments": sorted(self.markets)},
            "grades": {"by_count_pct": by_count, "by_notional_pct": by_notional,
                       "label": " · ".join(f"{g} {by_notional[g]:g}%" for g in "ABC" if by_notional.get(g))},
            "assumptions": {
                "latency_ms": cfg.latency_ms, "max_slippage": dstr(cfg.fill.max_slippage, None),
                "depth_penalty_bps": dstr(cfg.fill.depth_penalty / BP, None),
                "grade_c": {"half_spread_bps": dstr(cfg.fill.c_half_spread_bps, None),
                            "impact_bps_per_pct_volume": dstr(cfg.fill.c_impact_bps_per_pct, None)},
                "fees": {v: {"taker": dstr(self.fee_schedule(v).taker, None),
                             "maker": dstr(self.fee_schedule(v).maker, None)}
                         for v in sorted({s.fee_key for s in cfg.instruments.values()})},
                "trigger_by": dict(cfg.trigger_by), "margin_mode": "cross",
                "funding_price": {"hyperliquid": "oracle(index)", "other": "mark"},
                "liquidation": {"hl_partial_over_notional": dstr(cfg.hl_partial_liq_notional, None),
                                "taker_fee_on_liquidation": True,
                                "clearance_fee": {k: dstr(D(v), None) for k, v in cfg.liq_clearance_fee.items()}},
                "mmr_estimated": sorted(i for i, s in cfg.instruments.items() if s.mmr_estimated),
            },
            "flags": dict(sorted(self.flags.items())),
            "sources": cfg.sources,
            "internal_only_used": any(s.get("internal_only") for s in cfg.sources),
        }
        return RunResult(self.run_id, [o.to_row(self.run_id) for o in self.orders], list(self.fills),
                         list(self.ledger), metrics, meta, list(self.equity_curve))

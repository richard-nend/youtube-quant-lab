"""리스크 관리자 핵심 (빌드 계획 T26, 명세 execution-security-risk.md 4절). **주문하지 않는다** — 판정만.

서버 실행기·페이퍼·PC 모듈이 같은 코드를 쓴다(외부 의존 없음). 세 가지를 구분한다:
- 시스템 안전 장치(kind="system"): 사용자가 바꿀 수 없음 — 긴급 정지·정지 상태, 실거래 국가 차단, 관문 미통과 전략,
  AI 실거래 권한 단계, idempotency 중복·unknown 주문 재전송 금지, 전략별 초당 주문 수, 오래된 시세.
- 사용자 설정(kind="user_limit", `RiskLimits`): 기본값만 제공, 사용자가 올리기·내리기·끄기(None) 가능.
  완화는 웹 + 2단계 인증, 강화는 어디서나(MCP 포함) — `limits_change`.
- 경고(warnings): 주문을 막지 않음 — 청산 거리 경고, 가격 변동으로 한도를 넘은 상태 등.

판정은 사실 문장(fact)·값·한도를 함께 돌려준다(사실 원칙). 청산가·레버리지는 order_preview 와 같은 정의.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
from decimal import Decimal

from .events import BookEvent
from .instruments import InstrumentSpec
from .preview import order_preview
from .util import ZERO, D, dstr

RISK_VERSION = "0.1.0"

# ---- 시스템 상수 (사용자 설정 아님) ----
STALE_MARK_MS = 10_000          # 마크가 이보다 오래되면 신규 주문 중지
STALE_BOOK_MS = 10_000          # 시장가는 호가도 이 안이어야
MAX_ORDERS_PER_SEC = 5          # 같은 전략의 초당 주문 수
CONFIRM_TTL_MS = 60_000         # confirm_each 승인 대기 만료
BOOK_LEVELS = 20                # 호가 비율 기준 depth
IDEMPOTENCY_MEMORY = 100_000    # 기억하는 키 수(넘으면 오래된 것부터 잊음 — 원장의 유일 제약이 최종 방어)

AI_PERMISSIONS = ("off", "confirm_each", "auto_within_limits")


@dataclass(frozen=True)
class RiskLimits:
    """사용자 설정. None = 끔. 퍼센트는 0~100 숫자. 기본값: 명세 4절(명목가 $5k·일손실 5%·청산 거리 30%·
    경고 15%·호가 25%) + 어시스턴트 초안(레버리지 5배·최대 낙폭 20%·가격 이탈 5% — 인터페이스 v1 25절)."""
    max_account_notional: Decimal | None = Decimal(5000)
    max_instrument_notional: Decimal | None = None
    max_leverage: Decimal | None = Decimal(5)
    daily_loss_pct: Decimal | None = Decimal(5)
    max_drawdown_pct: Decimal | None = Decimal(20)
    min_liq_distance_pct: Decimal | None = Decimal(30)
    liq_warn_distance_pct: Decimal | None = Decimal(15)
    max_book_fraction: Decimal | None = Decimal("0.25")
    price_band_pct: Decimal | None = Decimal(5)
    on_stop: str = "halt"               # halt | halt_and_flatten (일손실·낙폭 도달 때)
    on_liq_warn: str = "warn"           # warn | warn_and_reduce

    def __post_init__(self):
        for f in fields(self):
            v = getattr(self, f.name)
            if f.type.startswith("Decimal") and v is not None:
                object.__setattr__(self, f.name, D(v))
        if self.on_stop not in ("halt", "halt_and_flatten") or self.on_liq_warn not in ("warn", "warn_and_reduce"):
            raise ValueError("on_stop: halt|halt_and_flatten, on_liq_warn: warn|warn_and_reduce")


# 완화 방향: 값이 커지면 완화(+1), 작아지면 완화(-1). None(끔)은 언제나 가장 느슨함.
_LOOSER = {"max_account_notional": 1, "max_instrument_notional": 1, "max_leverage": 1, "daily_loss_pct": 1,
           "max_drawdown_pct": 1, "min_liq_distance_pct": -1, "liq_warn_distance_pct": -1, "max_book_fraction": 1,
           "price_band_pct": 1}
_ORDER = {"on_stop": ["halt_and_flatten", "halt"], "on_liq_warn": ["warn_and_reduce", "warn"]}   # 뒤가 느슨함


@dataclass(frozen=True)
class OrderIntent:
    instrument_id: str
    side: str                           # buy | sell
    qty: Decimal
    idempotency_key: str
    origin: str = "user"                # user | ai | strategy
    strategy_id: str | None = None
    order_type: str = "market"          # market | limit
    limit_price: Decimal | None = None
    reduce_only: bool = False


@dataclass(frozen=True)
class MarketView:
    spec: InstrumentSpec
    mark: Decimal
    mark_ts_ms: int
    book: BookEvent | None = None
    book_ts_ms: int | None = None


@dataclass(frozen=True)
class AccountView:
    """판정 시점 계좌. positions = 상품별 수량(롱 +, 숏 −), 평가는 markets 의 마크."""
    equity: Decimal
    day_start_equity: Decimal           # UTC 0시 계좌 자산
    peak_equity: Decimal                # 최대 낙폭 기준 고점
    positions: dict[str, Decimal] = field(default_factory=dict)
    mode: str = "paper"                 # paper | live
    ai_permission: str = "off"
    live_country_blocked: bool = False  # 국가 판정(E1) 결과 — 실거래만 막는다
    gate_passed: dict[str, bool] = field(default_factory=dict)   # 전략별 실거래 관문 통과(서버 실행)


@dataclass
class Reason:
    code: str
    kind: str                           # system | user_limit
    fact: str
    value: str | None = None
    limit: str | None = None


@dataclass
class Decision:
    status: str                         # allow | reject | needs_confirmation
    reasons: list[Reason] = field(default_factory=list)
    warnings: list[Reason] = field(default_factory=list)
    confirm_expires_ms: int | None = None
    metrics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"risk_version": RISK_VERSION, "status": self.status, "reasons": [vars(r) for r in self.reasons],
                "warnings": [vars(w) for w in self.warnings], "confirm_expires_ms": self.confirm_expires_ms,
                "metrics": self.metrics}


@dataclass
class Action:
    """감시 결과 실행기가 할 일: halt(신규 중지) · cancel_open · flatten · reduce · alert."""
    type: str
    scope: tuple[str, str | None]
    reason: Reason


def _pct(x: Decimal) -> str:
    return dstr(x, Decimal("0.01"))


class RiskState:
    """정지·긴급 정지·주문 기억. 실행기가 프로세스 하나에 하나(재시작 때 원장·감사 로그에서 복원)."""

    def __init__(self):
        self.halts: dict[tuple[str, str | None], Reason] = {}       # 리스크 관리자가 건 정지
        self.kills: dict[tuple[str, str | None], str] = {}          # 긴급 정지: scope → 발동 주체
        self.keys: dict[str, str] = {}                              # idempotency_key → 주문 상태
        self.recent: dict[str, list[int]] = {}                      # strategy_id → 최근 주문 시각

    # ---- 긴급 정지 (전체·전략·종목) ----
    def engage_kill(self, scope: tuple[str, str | None], by: str, flatten: bool = False) -> list[Action]:
        """by: user | risk_manager | operator. 모든 미체결 취소, flatten 이면 포지션 정리."""
        if scope[0] not in ("all", "strategy", "instrument"):
            raise ValueError("scope: all | strategy | instrument")
        self.kills[scope] = by
        r = Reason("kill_switch", "system", f"kill switch engaged for {scope[0]} {scope[1] or ''}".strip() + f" by {by}")
        return [Action("cancel_open", scope, r)] + ([Action("flatten", scope, r)] if flatten else [])

    def release_kill(self, scope: tuple[str, str | None], via: str, actor: str) -> None:
        """해제는 웹에서 사용자만(명세 4절)."""
        if not (via == "web" and actor == "user"):
            raise PermissionError("kill switch can be released only by the user on the web")
        self.kills.pop(scope, None)

    def release_halt(self, scope: tuple[str, str | None], via: str, actor: str) -> None:
        if not (via == "web" and actor == "user"):
            raise PermissionError("risk halts can be released only by the user on the web")
        self.halts.pop(scope, None)

    def blocking(self, intent: OrderIntent) -> Reason | None:
        scopes = [("all", None), ("instrument", intent.instrument_id)] + ([("strategy", intent.strategy_id)] if intent.strategy_id else [])
        for s in scopes:
            if s in self.kills:
                return Reason("kill_switch", "system", f"kill switch is engaged for {s[0]} {s[1] or ''}".strip())
            if s in self.halts:
                h = self.halts[s]
                return Reason("halted", "system", f"new orders are halted for {s[0]} {s[1] or ''}: {h.fact}".replace("  ", " "))
        return None

    # ---- 주문 기억 ----
    def record(self, key: str, order_state: str) -> None:
        """실행기가 주문 상태가 바뀔 때마다 알린다(created·submitted·unknown·filled…)."""
        self.keys[key] = order_state
        if len(self.keys) > IDEMPOTENCY_MEMORY:
            del self.keys[next(iter(self.keys))]

    def _rate(self, strategy_id: str | None, now_ms: int) -> int:
        if not strategy_id:
            return 0
        ts = [t for t in self.recent.get(strategy_id, []) if now_ms - t < 1000]
        self.recent[strategy_id] = ts
        return len(ts)


def _book_depth(book: BookEvent | None, side: str) -> Decimal | None:
    if book is None:
        return None
    lv = book.asks if side == "buy" else book.bids
    return sum((q for _, q in lv[:BOOK_LEVELS]), ZERO) if lv else None


def _account_notional(acct: AccountView, markets: dict[str, MarketView], override: tuple[str, Decimal] | None = None) -> Decimal:
    pos = dict(acct.positions)
    if override:
        pos[override[0]] = override[1]
    total = ZERO
    for iid, q in pos.items():
        m = markets.get(iid)
        if m is not None and q:
            total += abs(q) * m.mark
    return total


def _other_maintenance(acct: AccountView, markets: dict[str, MarketView], skip: str) -> Decimal:
    total = ZERO
    for iid, q in acct.positions.items():
        m = markets.get(iid)
        if iid != skip and q and m is not None and not m.spec.is_spot:
            total += m.spec.maintenance(q, m.mark)
    return total


def check_order(intent: OrderIntent, acct: AccountView, markets: dict[str, MarketView], limits: RiskLimits,
                state: RiskState, now_ms: int) -> Decision:
    """주문 전 검사. 걸린 사유를 모두 모아 돌려준다(첫 사유에서 멈추지 않음 — AI·사용자가 한 번에 고치게)."""
    reasons: list[Reason] = []
    warns: list[Reason] = []
    qty = D(intent.qty)
    mkt = markets.get(intent.instrument_id)
    if intent.side not in ("buy", "sell") or qty <= 0:
        return Decision("reject", [Reason("invalid_order", "system", "side must be buy|sell and qty > 0")])
    if mkt is None:
        return Decision("reject", [Reason("no_market", "system", f"no market data for {intent.instrument_id}")])
    live = acct.mode == "live"

    # ---- 시스템 안전 장치 ----
    b = state.blocking(intent)
    if b and not intent.reduce_only:          # 정지 중에도 줄이는 주문은 허용(정리용)
        reasons.append(b)
    if live and acct.live_country_blocked:
        reasons.append(Reason("country_blocked", "system", "live trading is not available in your country"))
    if live and intent.origin == "strategy" and intent.strategy_id and not acct.gate_passed.get(intent.strategy_id, False):
        reasons.append(Reason("gate_not_passed", "system", f"strategy {intent.strategy_id} has not passed the pre-live gate"))
    if live and intent.origin == "ai" and acct.ai_permission == "off":
        reasons.append(Reason("ai_permission_off", "system", "AI live orders are off for this account (change on the web)"))
    prev = state.keys.get(intent.idempotency_key)
    if prev == "unknown":
        reasons.append(Reason("unknown_order_pending", "system",
                              "an order with this idempotency key has an unknown result; it is never resent — status query only"))
    elif prev is not None:
        reasons.append(Reason("duplicate_order", "system", f"idempotency key already used (order state {prev})"))
    n = state._rate(intent.strategy_id, now_ms)
    if n >= MAX_ORDERS_PER_SEC:
        reasons.append(Reason("order_rate", "system", f"strategy sent {n} orders in the last second", str(n), str(MAX_ORDERS_PER_SEC)))
    age = now_ms - mkt.mark_ts_ms
    if age > STALE_MARK_MS:
        reasons.append(Reason("stale_market_data", "system", f"mark price is {age / 1000:.1f}s old", f"{age}", f"{STALE_MARK_MS}"))
    if intent.order_type == "market" and (mkt.book is None or mkt.book_ts_ms is None or now_ms - mkt.book_ts_ms > STALE_BOOK_MS):
        reasons.append(Reason("stale_market_data", "system", "market orders need an order book snapshot from the last 10s"))

    # ---- 손실 정지 상태(감시가 정지를 걸기 전이라도 주문 시점에 다시 본다) ----
    loss_pct = (acct.day_start_equity - acct.equity) / acct.day_start_equity * 100 if acct.day_start_equity > 0 else ZERO
    dd_pct = (acct.peak_equity - acct.equity) / acct.peak_equity * 100 if acct.peak_equity > 0 else ZERO
    metrics: dict = {"daily_loss_pct": _pct(loss_pct), "drawdown_pct": _pct(dd_pct)}
    if not intent.reduce_only:
        if limits.daily_loss_pct is not None and loss_pct >= limits.daily_loss_pct:
            reasons.append(Reason("daily_loss_stop", "user_limit", f"today's loss {_pct(loss_pct)}% reached the daily loss limit",
                                  _pct(loss_pct), _pct(limits.daily_loss_pct)))
        if limits.max_drawdown_pct is not None and dd_pct >= limits.max_drawdown_pct:
            reasons.append(Reason("max_drawdown_stop", "user_limit", f"drawdown from peak {_pct(dd_pct)}% reached the limit",
                                  _pct(dd_pct), _pct(limits.max_drawdown_pct)))

    # ---- 사용자 설정: 가격·규모 ----
    if intent.order_type == "limit":
        if intent.limit_price is None:
            return Decision("reject", [Reason("invalid_order", "system", "limit order needs limit_price")])
        band = abs(D(intent.limit_price) - mkt.mark) / mkt.mark * 100
        metrics["limit_vs_mark_pct"] = _pct(band)
        if limits.price_band_pct is not None and band > limits.price_band_pct:
            reasons.append(Reason("price_band", "user_limit", f"limit price is {_pct(band)}% away from mark",
                                  _pct(band), _pct(limits.price_band_pct)))
    depth = _book_depth(mkt.book, intent.side)
    if depth is not None and depth > 0:
        frac = qty / depth
        metrics["book_fraction"] = dstr(frac, Decimal("0.0001"))
        if limits.max_book_fraction is not None and frac > limits.max_book_fraction and not intent.reduce_only:
            reasons.append(Reason("book_depth", "user_limit", f"order is {_pct(frac * 100)}% of the top {BOOK_LEVELS} levels on its side",
                                  dstr(frac, Decimal("0.0001")), dstr(limits.max_book_fraction)))

    sgn = 1 if intent.side == "buy" else -1
    q0 = acct.positions.get(intent.instrument_id, ZERO)
    q1 = q0 + sgn * qty
    if intent.reduce_only and (q0 == 0 or abs(q1) > abs(q0) or (q1 != 0 and (q1 > 0) != (q0 > 0))):
        reasons.append(Reason("reduce_only_increases", "system", "reduce-only order would open or flip the position"))
    inst_notional = abs(q1) * mkt.mark
    acct_notional = _account_notional(acct, markets, (intent.instrument_id, q1))
    metrics.update(instrument_notional_after=dstr(inst_notional, Decimal("0.01")), account_notional_after=dstr(acct_notional, Decimal("0.01")))
    increasing = abs(q1) > abs(q0)
    if increasing:
        if limits.max_instrument_notional is not None and inst_notional > limits.max_instrument_notional:
            reasons.append(Reason("max_instrument_notional", "user_limit", f"{intent.instrument_id} notional after the order would be {dstr(inst_notional, Decimal('0.01'))}",
                                  dstr(inst_notional, Decimal("0.01")), dstr(limits.max_instrument_notional)))
        if limits.max_account_notional is not None and acct_notional > limits.max_account_notional:
            reasons.append(Reason("max_account_notional", "user_limit", f"account notional after the order would be {dstr(acct_notional, Decimal('0.01'))}",
                                  dstr(acct_notional, Decimal("0.01")), dstr(limits.max_account_notional)))
        lev = acct_notional / acct.equity if acct.equity > 0 else None
        metrics["account_leverage_after"] = dstr(lev, Decimal("0.001")) if lev is not None else None
        if limits.max_leverage is not None and (lev is None or lev > limits.max_leverage):
            reasons.append(Reason("max_leverage", "user_limit", f"account leverage after the order would be {dstr(lev, Decimal('0.01')) if lev is not None else 'unbounded'}x",
                                  dstr(lev, Decimal("0.001")) if lev is not None else None, dstr(limits.max_leverage)))
        # 청산 거리: order_preview 와 같은 정의(교차, 다른 포지션 가격 고정)
        if not mkt.spec.is_spot:
            try:
                pv = order_preview(mkt.spec, intent.side, qty, mark=mkt.mark, equity=acct.equity, book=mkt.book,
                                   order_type=intent.order_type, limit_price=intent.limit_price, position_qty=q0,
                                   other_maintenance=_other_maintenance(acct, markets, intent.instrument_id))
                dist = pv["liquidation_distance_pct"]
                metrics["liquidation_distance_pct_after"] = dist
                if "below_maintenance_after_fill" in pv["flags"]:
                    reasons.append(Reason("below_maintenance", "system", "account would be below maintenance margin right after the fill"))
                elif dist is not None:
                    d = D(dist)
                    if limits.min_liq_distance_pct is not None and d < limits.min_liq_distance_pct:
                        reasons.append(Reason("min_liq_distance", "user_limit", f"liquidation would be {_pct(d)}% from mark",
                                              _pct(d), _pct(limits.min_liq_distance_pct)))
                    elif limits.liq_warn_distance_pct is not None and d < limits.liq_warn_distance_pct:
                        warns.append(Reason("liq_distance_warning", "user_limit", f"liquidation would be {_pct(d)}% from mark",
                                            _pct(d), _pct(limits.liq_warn_distance_pct)))
            except ValueError as e:
                reasons.append(Reason("preview_failed", "system", f"pre-trade calculation failed: {e}"))

    if reasons:
        return Decision("reject", reasons, warns, metrics=metrics)
    # 통과: 기억(같은 키 재사용 방지)·초당 횟수에 넣는다
    state.record(intent.idempotency_key, "created")
    if intent.strategy_id:
        state.recent.setdefault(intent.strategy_id, []).append(now_ms)
    if live and intent.origin == "ai" and acct.ai_permission == "confirm_each":
        return Decision("needs_confirmation", [], warns, now_ms + CONFIRM_TTL_MS, metrics)
    return Decision("allow", [], warns, metrics=metrics)


def monitor(acct: AccountView, markets: dict[str, MarketView], limits: RiskLimits, state: RiskState, now_ms: int) -> list[Action]:
    """실행 중 감시(마크 갱신마다). 같은 정지를 두 번 걸지 않는다."""
    actions: list[Action] = []
    acct_scope = ("all", None)

    def stop(code: str, fact: str, value: Decimal, limit: Decimal):
        if acct_scope in state.halts:
            return
        r = Reason(code, "user_limit", fact, _pct(value), _pct(limit))
        state.halts[acct_scope] = r
        actions.append(Action("halt", acct_scope, r))
        if limits.on_stop == "halt_and_flatten":
            actions.extend([Action("cancel_open", acct_scope, r), Action("flatten", acct_scope, r)])

    if acct.day_start_equity > 0 and limits.daily_loss_pct is not None:
        loss = (acct.day_start_equity - acct.equity) / acct.day_start_equity * 100
        if loss >= limits.daily_loss_pct:
            stop("daily_loss_stop", f"today's loss {_pct(loss)}% reached the daily loss limit", loss, limits.daily_loss_pct)
    if acct.peak_equity > 0 and limits.max_drawdown_pct is not None:
        dd = (acct.peak_equity - acct.equity) / acct.peak_equity * 100
        if dd >= limits.max_drawdown_pct:
            stop("max_drawdown_stop", f"drawdown from peak {_pct(dd)}% reached the limit", dd, limits.max_drawdown_pct)

    for iid, q in acct.positions.items():
        m = markets.get(iid)
        if m is None or not q:
            continue
        if now_ms - m.mark_ts_ms > STALE_MARK_MS:
            actions.append(Action("alert", ("instrument", iid), Reason("stale_market_data", "system",
                                  f"{iid} mark price is {(now_ms - m.mark_ts_ms) / 1000:.1f}s old; new orders stop until it is fresh")))
            continue
        if m.spec.is_spot or limits.liq_warn_distance_pct is None:
            continue
        l = m.spec.maint_rate()
        den = q - abs(q) * l
        if den == 0:
            continue
        liq = (_other_maintenance(acct, markets, iid) - m.spec.maint_amount - acct.equity + q * m.mark) / den
        if liq <= 0:
            continue
        dist = abs(liq - m.mark) / m.mark * 100
        if dist < limits.liq_warn_distance_pct:
            r = Reason("liq_distance_warning", "user_limit", f"{iid} liquidation is {_pct(dist)}% from mark", _pct(dist), _pct(limits.liq_warn_distance_pct))
            actions.append(Action("alert", ("instrument", iid), r))
            if limits.on_liq_warn == "warn_and_reduce":
                actions.append(Action("reduce", ("instrument", iid), r))

    if limits.max_account_notional is not None:
        n = _account_notional(acct, markets)
        if n > limits.max_account_notional:
            actions.append(Action("alert", acct_scope, Reason("over_account_notional", "user_limit",
                                  f"account notional {dstr(n, Decimal('0.01'))} is above the limit after price moves (new increases are rejected)",
                                  dstr(n, Decimal("0.01")), dstr(limits.max_account_notional))))
    return actions


def _looser(name: str, old, new) -> bool:
    if name in _ORDER:
        return _ORDER[name].index(new) > _ORDER[name].index(old)
    if old is None:
        return False
    if new is None:
        return True
    return (D(new) - D(old)) * _LOOSER[name] > 0


def limits_change(current: RiskLimits, changes: dict, via: str, actor: str, mfa: bool = False) -> tuple[RiskLimits, list[str]]:
    """한도 바꾸기. 강화는 어디서나(MCP·웹·AI), 완화는 웹 + 사용자 + 2단계 인증만. 반환: (새 한도, 완화된 항목)."""
    unknown = set(changes) - {f.name for f in fields(RiskLimits)}
    if unknown:
        raise ValueError(f"unknown limits: {sorted(unknown)}")
    loosened = [k for k, v in changes.items() if _looser(k, getattr(current, k), v)]
    if loosened and not (via == "web" and actor == "user" and mfa):
        raise PermissionError(f"loosening {loosened} needs the user on the web with two-factor authentication")
    return replace(current, **changes), loosened


def ai_permission_change(current: str, new: str, via: str, actor: str, mfa: bool = False) -> str:
    """AI 실거래 권한: 낮추기는 어디서나, 올리기는 웹 + 사용자 + 2단계 인증. AI는 스스로 올리지 못한다."""
    if new not in AI_PERMISSIONS:
        raise ValueError(f"ai_permission: {AI_PERMISSIONS}")
    if AI_PERMISSIONS.index(new) > AI_PERMISSIONS.index(current) and not (via == "web" and actor == "user" and mfa):
        raise PermissionError("raising AI permission needs the user on the web with two-factor authentication")
    return new

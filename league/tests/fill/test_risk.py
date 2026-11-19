from __future__ import annotations

from decimal import Decimal

import pytest

from league.fill import BookEvent, InstrumentSpec
from league.fill.risk import (CONFIRM_TTL_MS, MAX_ORDERS_PER_SEC, AccountView, MarketView, OrderIntent, RiskLimits,
                                   RiskState, ai_permission_change, check_order, limits_change, monitor)

NOW = 1_800_000_000_000
BTC = InstrumentSpec("BTC-USDC-PERP", "hyperliquid", max_leverage=40)
ETH = InstrumentSpec("ETH-USDC-PERP", "hyperliquid", max_leverage=25)
BOOK = BookEvent(NOW, BTC.instrument_id, [(99_990, 1), (99_980, 1)], [(100_010, 1), (100_020, 1)])


def mk(mark=100_000, ts=NOW, book=BOOK, spec=BTC):
    return MarketView(spec, Decimal(mark), ts, book, ts if book else None)


def acct(**kw):
    base = dict(equity=Decimal(10_000), day_start_equity=Decimal(10_000), peak_equity=Decimal(10_000))
    base.update(kw)
    return AccountView(**base)


def order(qty="0.01", key="k1", **kw):
    return OrderIntent(BTC.instrument_id, kw.pop("side", "buy"), Decimal(qty), key, **kw)


def codes(d):
    return [r.code for r in d.reasons]


def test_small_order_allowed_and_key_remembered():
    st = RiskState()
    d = check_order(order(), acct(), {BTC.instrument_id: mk()}, RiskLimits(), st, NOW)
    assert d.status == "allow", d.to_dict()
    assert d.metrics["account_notional_after"] == "1000"
    again = check_order(order(), acct(), {BTC.instrument_id: mk()}, RiskLimits(), st, NOW)
    assert codes(again) == ["duplicate_order"]
    st.record("k2", "unknown")
    assert codes(check_order(order(key="k2"), acct(), {BTC.instrument_id: mk()}, RiskLimits(), st, NOW)) == ["unknown_order_pending"]


def test_user_limits_collect_all_reasons_with_facts():
    # 0.9 BTC = $90k: 명목가 $5k·레버리지 5배·호가 25%(같은 쪽 2 BTC) 모두 넘음
    d = check_order(order("0.9"), acct(), {BTC.instrument_id: mk()}, RiskLimits(), RiskState(), NOW)
    assert d.status == "reject"
    assert {"max_account_notional", "max_leverage", "book_depth"} <= set(codes(d))
    r = next(x for x in d.reasons if x.code == "max_account_notional")
    assert r.kind == "user_limit" and r.limit == "5000" and "notional" in r.fact
    # 사용자가 끄면(None) 통과 — 청산 거리만 남는지 확인
    off = RiskLimits(max_account_notional=None, max_leverage=None, max_book_fraction=None)
    d2 = check_order(order("0.9"), acct(), {BTC.instrument_id: mk()}, off, RiskState(), NOW)
    assert "max_account_notional" not in codes(d2) and "book_depth" not in codes(d2)


def test_liquidation_distance_reject_and_warning():
    loose = RiskLimits(max_account_notional=None, max_leverage=None, max_book_fraction=None)
    # 자산 $10k로 0.5 BTC($50k, 5배) → 청산 거리 약 19% < 30% → 거절
    d = check_order(order("0.5"), acct(), {BTC.instrument_id: mk()}, loose, RiskState(), NOW)
    assert "min_liq_distance" in codes(d)
    dist = Decimal(d.metrics["liquidation_distance_pct_after"])
    assert 15 < dist < 30
    # 최소값을 끄고 경고 기준을 25%로 두면 주문은 통과하고 경고만
    d2 = check_order(order("0.5", key="k3"), acct(), {BTC.instrument_id: mk()},
                     RiskLimits(max_account_notional=None, max_leverage=None, max_book_fraction=None,
                                min_liq_distance_pct=None, liq_warn_distance_pct=25), RiskState(), NOW)
    assert d2.status == "allow" and [w.code for w in d2.warnings] == ["liq_distance_warning"]


def test_system_guards():
    m = {BTC.instrument_id: mk()}
    live = acct(mode="live", live_country_blocked=True)
    assert "country_blocked" in codes(check_order(order(), live, m, RiskLimits(), RiskState(), NOW))
    assert "country_blocked" not in codes(check_order(order(), acct(live_country_blocked=True), m, RiskLimits(), RiskState(), NOW))  # 페이퍼는 허용
    assert "gate_not_passed" in codes(check_order(order(origin="strategy", strategy_id="s1"), acct(mode="live"), m, RiskLimits(), RiskState(), NOW))
    assert "ai_permission_off" in codes(check_order(order(origin="ai"), acct(mode="live"), m, RiskLimits(), RiskState(), NOW))
    c = check_order(order(origin="ai"), acct(mode="live", ai_permission="confirm_each"), m, RiskLimits(), RiskState(), NOW)
    assert c.status == "needs_confirmation" and c.confirm_expires_ms == NOW + CONFIRM_TTL_MS
    stale = {BTC.instrument_id: mk(ts=NOW - 11_000)}
    assert "stale_market_data" in codes(check_order(order(), acct(), stale, RiskLimits(), RiskState(), NOW))
    nobook = {BTC.instrument_id: mk(book=None)}
    assert "stale_market_data" in codes(check_order(order(), acct(), nobook, RiskLimits(), RiskState(), NOW))
    # 가격 이탈: 지정가가 마크에서 10% → 5% 기준 거절(호가 없이도 지정가는 가능)
    lim = order(order_type="limit", limit_price=Decimal(110_000))
    assert codes(check_order(lim, acct(), nobook, RiskLimits(), RiskState(), NOW)) == ["price_band"]
    # 전략 초당 주문 수
    st = RiskState()
    for i in range(MAX_ORDERS_PER_SEC):
        assert check_order(order(key=f"r{i}", strategy_id="s"), acct(), m, RiskLimits(), st, NOW + i).status == "allow"
    assert "order_rate" in codes(check_order(order(key="r9", strategy_id="s"), acct(), m, RiskLimits(), st, NOW + 10))
    assert check_order(order(key="r10", strategy_id="s"), acct(), m, RiskLimits(), st, NOW + 1500).status == "allow"


def test_kill_switch_and_halts_allow_reduce_only():
    st = RiskState()
    acts = st.engage_kill(("instrument", BTC.instrument_id), by="user", flatten=True)
    assert [a.type for a in acts] == ["cancel_open", "flatten"]
    m = {BTC.instrument_id: mk()}
    held = acct(positions={BTC.instrument_id: Decimal("0.02")})
    assert codes(check_order(order(), held, m, RiskLimits(), st, NOW)) == ["kill_switch"]
    assert check_order(order(side="sell", reduce_only=True, key="r"), held, m, RiskLimits(), st, NOW).status == "allow"
    assert "reduce_only_increases" in codes(check_order(order("0.05", side="sell", reduce_only=True, key="r2"), held, m, RiskLimits(), st, NOW))
    with pytest.raises(PermissionError):
        st.release_kill(("instrument", BTC.instrument_id), via="mcp", actor="user")
    with pytest.raises(PermissionError):
        st.release_kill(("instrument", BTC.instrument_id), via="web", actor="ai")
    st.release_kill(("instrument", BTC.instrument_id), via="web", actor="user")
    assert check_order(order(key="k9"), held, m, RiskLimits(), st, NOW).status == "allow"


def test_monitor_daily_loss_drawdown_liq_warning():
    st = RiskState()
    m = {BTC.instrument_id: mk(), ETH.instrument_id: mk(3_000, spec=ETH, book=None)}
    losing = acct(equity=Decimal(9_400), positions={BTC.instrument_id: Decimal("0.01")})
    acts = monitor(losing, m, RiskLimits(on_stop="halt_and_flatten"), st, NOW)
    assert [a.type for a in acts][:3] == ["halt", "cancel_open", "flatten"]
    assert acts[0].reason.code == "daily_loss_stop" and acts[0].reason.value == "6"
    assert monitor(losing, m, RiskLimits(), st, NOW) == []                   # 같은 정지는 한 번
    assert codes(check_order(order(key="x"), losing, m, RiskLimits(), st, NOW))[0] == "halted"
    with pytest.raises(PermissionError):
        st.release_halt(("all", None), via="mcp", actor="user")
    # 청산 거리 경고 + 축소, 가격 변동으로 명목가 초과 알림
    st2 = RiskState()
    big = acct(equity=Decimal(10_000), positions={BTC.instrument_id: Decimal("0.6")})
    acts2 = monitor(big, m, RiskLimits(daily_loss_pct=None, liq_warn_distance_pct=20, on_liq_warn="warn_and_reduce"), st2, NOW)
    assert ("alert", "liq_distance_warning") in [(a.type, a.reason.code) for a in acts2]
    assert "reduce" in [a.type for a in acts2]
    assert "over_account_notional" in [a.reason.code for a in acts2]
    stale = {BTC.instrument_id: mk(ts=NOW - 60_000)}
    assert [a.reason.code for a in monitor(acct(positions={BTC.instrument_id: Decimal("0.01")}), stale, RiskLimits(), RiskState(), NOW)] == ["stale_market_data"]


def test_limit_and_permission_changes():
    cur = RiskLimits()
    new, loosened = limits_change(cur, {"max_account_notional": 2000, "min_liq_distance_pct": 40}, via="mcp", actor="ai")
    assert new.max_account_notional == 2000 and loosened == []          # 강화는 MCP·AI도 가능
    for ch in ({"max_account_notional": 20_000}, {"min_liq_distance_pct": 10}, {"daily_loss_pct": None}, {"on_stop": "halt"}):
        base = RiskLimits(on_stop="halt_and_flatten")
        with pytest.raises(PermissionError):
            limits_change(base, ch, via="mcp", actor="user")
        with pytest.raises(PermissionError):
            limits_change(base, ch, via="web", actor="user", mfa=False)
        _, lo = limits_change(base, ch, via="web", actor="user", mfa=True)
        assert lo == list(ch)
    with pytest.raises(ValueError):
        limits_change(cur, {"nope": 1}, via="web", actor="user", mfa=True)
    assert ai_permission_change("auto_within_limits", "off", via="mcp", actor="ai") == "off"
    with pytest.raises(PermissionError):
        ai_permission_change("off", "confirm_each", via="mcp", actor="ai")
    assert ai_permission_change("off", "confirm_each", via="web", actor="user", mfa=True) == "confirm_each"

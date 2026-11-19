from decimal import Decimal

from league.core import League
from league.decision import applied_ids, mark_applied, to_orders, validate
from league.fill import BookEvent, OrderRequest
from league.fill.engine import Cancel
from league.venues import BTC, ETH

MARK = {"BTC": Decimal(100), "ETH": Decimal(100)}


def item(sym="BTC", action="long", pct=100, order="market", **kw):
    return {"symbol": sym, "action": action, "notional_pct": pct, "order": order, "reason": "r", **kw}


def dec(items, utc="20261119T0000", status="ok"):
    return {"team": "claude", "decided_utc": utc, "status": status, "items": items}


def league(pos_qty=0, rest_bid=None):
    lg = League(["claude"], 0)
    for i in range(3):
        for iid in (BTC, ETH):
            lg.step(BookEvent(i * 1000, iid, [(Decimal(99), Decimal(500))], [(Decimal(100), Decimal(500))]))
        if i == 0 and pos_qty:
            lg.submit("claude", [OrderRequest(BTC, "buy" if pos_qty > 0 else "sell", "market", abs(Decimal(pos_qty)))])
        if i == 1 and rest_bid:
            lg.submit("claude", [OrderRequest(BTC, "buy", "limit", Decimal(1), price=Decimal(50), tag="claude:old")])
    return lg


def test_scale_to_leverage_cap():
    v = league().view("claude")
    o = to_orders(dec([item("BTC", "long", 250), item("ETH", "long", 200)]), v, MARK)
    q = {x.instrument_id: x.qty for x in o}
    eq = v.equity()
    assert q[BTC] == (eq * Decimal(300) * 250 / 450 / 100 / 100).quantize(Decimal("0.001"), "ROUND_DOWN")
    assert abs(q[BTC] * 100 / eq - Decimal("1.6667")) < Decimal("0.001")
    assert abs(q[ETH] * 100 / eq - Decimal("1.3333")) < Decimal("0.001")


def test_hold_keeps_position_and_orders():
    v = league(rest_bid=True).view("claude")
    assert v.open_orders(BTC)
    assert to_orders(dec([item(action="hold")]), v, MARK) == []
    assert to_orders(dec([item()], status="hold_fallback"), v, MARK) == []


def test_new_decision_cancels_previous_unfilled():
    v = league(rest_bid=True).view("claude")
    o = to_orders(dec([item(pct=10)]), v, MARK)
    old = v.open_orders(BTC)[0]
    assert Cancel(ord_id=old.ord_id) in o and old.req.tag == "claude:old"
    assert any(isinstance(x, OrderRequest) and x.type == "market" for x in o)


def test_stop_and_tp_are_reduce_only_triggers():
    o = to_orders(dec([item(stop=90, take_profit=120)]), league().view("claude"), MARK)
    st, tp = [x for x in o if x.type != "market"]
    assert (st.type, st.side, st.reduce_only, st.trigger.price, st.trigger.by) == ("stop_market", "sell", True, 90, "mark")
    assert (tp.type, tp.reduce_only, tp.trigger.price) == ("take_profit_market", True, 120)
    assert st.qty == o[0].qty


def test_close_flattens():
    v = league(pos_qty=3).view("claude")
    assert v.position(BTC) == 3
    (o,) = to_orders(dec([item(action="close", pct=0)]), v, MARK)
    assert (o.side, o.qty, o.reduce_only, o.type) == ("sell", 3, True, "market")


def test_flip_and_difference_only():
    v = league(pos_qty=3).view("claude")
    o = to_orders(dec([item(action="short", pct=100)]), v, MARK)
    assert [(x.side, x.reduce_only) for x in o] == [("sell", True), ("sell", False)]
    # 같은 방향이면 차이만
    big = to_orders(dec([item(pct=100)]), v, MARK)
    want = (v.equity() / 100).quantize(Decimal("0.001"), "ROUND_DOWN")
    assert big[0].qty == want - 3 and big[0].side == "buy"


def test_invalid_json_is_rejected_by_validate():
    assert validate([]) and validate({"team": "x"})
    assert validate(dec([item(action="yolo")])) and validate(dec([item(pct=301)]))
    assert validate(dec([item(order="limit")])) and validate(dec([item(sym="DOGE")]))
    assert validate(dec([item(), item("ETH", "short", 5, order="limit", limit_price=9)])) == []


def test_decision_applied_once(tmp_path):
    p, v, sent = tmp_path / "applied.txt", league().view("claude"), []
    for _ in range(2):
        if "f.json" not in applied_ids(p):
            sent += to_orders(dec([item()]), v, MARK)
            mark_applied(p, "f.json")
    assert len(sent) == 1


def test_cap_counts_untouched_existing_position():
    v = league(pos_qty=125).view("claude")   # BTC 약 250%
    o = to_orders(dec([item("ETH", "long", 300), item("BTC", "hold", 0)]), v, MARK)
    eth = sum(x.qty for x in o if x.instrument_id == ETH) * 100 / v.equity() * 100
    assert abs(eth - (300 - abs(v.position(BTC)) * 100 / v.equity() * 100)) < Decimal("0.2")
    assert 45 < eth < 55


def test_cap_held_override_counts_as_used():
    v = league(pos_qty=125).view("claude")
    o = to_orders(dec([item("BTC", "long", 300), item("ETH", "long", 200)]), v, MARK)
    btc = v.position(BTC) + sum(x.qty if x.side == "buy" else -x.qty for x in o if x.instrument_id == BTC)
    eth = sum(x.qty for x in o if x.instrument_id == ETH)
    tot = (btc + eth) * 100 / v.equity() * 100
    assert abs(btc - v.position(BTC)) < Decimal("0.01") and 45 < eth * 100 / v.equity() * 100 < 55 and tot <= 300


def test_tiny_pct_keeps_resting_orders():
    v = league(rest_bid=True).view("claude")
    assert v.open_orders(BTC)
    assert to_orders(dec([item(pct=0.0001)]), v, MARK) == []


def test_bad_equity_and_non_dict():
    assert validate(None) and validate("x")
    assert to_orders(None, league().view("claude"), MARK) == []

def test_cap_with_partial_untouched():
    v = league(pos_qty=25).view("claude")    # 25*100/5000 = 50%
    o = to_orders(dec([item("ETH", "long", 300), item("BTC", "hold", 0)]), v, MARK)
    eth = sum(x.qty for x in o if x.instrument_id == ETH)
    assert abs(eth * 100 / v.equity() * 100 - 250) < Decimal("0.2")


def test_malformed_returns_hold():
    v = league(rest_bid=True).view("claude")
    assert to_orders({"team": "claude"}, v, MARK) == []
    assert to_orders(dec([item(action="yolo")]), v, MARK) == []


def test_missing_or_zero_mark_is_hold():
    v = league().view("claude")
    assert to_orders(dec([item("ETH", "long", 10)]), v, {"BTC": Decimal(100)}) == []
    assert to_orders(dec([item()]), v, {"BTC": Decimal(0)}) == []


def test_tiny_pct_does_not_flatten():
    v = league(pos_qty=3).view("claude")
    assert to_orders(dec([item(pct=0.001)]), v, MARK) == []

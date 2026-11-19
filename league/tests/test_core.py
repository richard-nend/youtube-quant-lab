from decimal import Decimal

from league.core import TEAMS, League
from league.fill import BookEvent, OrderRequest, TradeEvent
from league.venues import BTC

BIDS = [(Decimal(99), Decimal(5))]
ASKS = [(Decimal(100), Decimal(5))]


def mkt(side="buy", qty="1"):
    return OrderRequest(BTC, side, "market", Decimal(qty))


def lim(side, qty, px):
    return OrderRequest(BTC, side, "limit", Decimal(qty), price=Decimal(px))


def events(n=40):
    out = []
    for i in range(n):
        t = i * 1000
        out.append(BookEvent(t, BTC, BIDS, ASKS))
        out.append(TradeEvent(t + 10, BTC, Decimal(100), Decimal(1), "buy"))
    return out


def run_league(evs, orders, lg=None):
    """orders: {event index: [(team, items)]} — 해당 이벤트 처리 전에 submit."""
    lg = lg or League(list(TEAMS), 0)
    for i, e in evs:
        for team, items in orders.get(i, []):
            lg.submit(team, items)
        lg.step(e)
    return lg


ORDERS = {10: [("claude", [mkt()])], 20: [("gpt", [lim("buy", "1", 98)]), ("ours", [mkt("sell", "2")])]}


def state(lg):
    return {t: (lg.view(t).equity(), lg.view(t).position(BTC), len(lg.view(t).open_orders()),
                len(lg.engines[t].fills)) for t in TEAMS}


def test_same_events_same_market_all_teams():
    lg = run_league(list(enumerate(events())), {})
    assert {lg.view(t).equity() for t in TEAMS} == {Decimal(5000)}


def test_submit_only_affects_that_team():
    lg = run_league(list(enumerate(events())), {5: [("claude", [mkt()])]})
    assert lg.view("claude").position(BTC) > 0
    assert all(lg.view(t).position(BTC) == 0 for t in TEAMS if t != "claude")


def test_snapshot_replay_equals_uninterrupted(tmp_path):
    evs = list(enumerate(events()))
    a = run_league(evs, ORDERS)
    k = 15
    b = run_league(evs[:k], {i: v for i, v in ORDERS.items() if i < k})
    b.snapshot(tmp_path / "s.pkl")
    c = League.restore(tmp_path / "s.pkl")
    assert c.last_ts == evs[k - 1][1].at_ms
    run_league(evs[k:], {i: v for i, v in ORDERS.items() if i >= k}, c)
    assert state(c) == state(a)
    assert any(state(a)[t][3] for t in TEAMS)  # 체결이 실제로 있었는지


def test_gap_downgrades_grade():
    evs = [BookEvent(0, BTC, BIDS, ASKS), TradeEvent(10, BTC, Decimal(100), Decimal(1), "buy"),
           BookEvent(1000, BTC, BIDS, ASKS), TradeEvent(1010, BTC, Decimal(100), Decimal(1), "buy"),
           BookEvent(11_000, BTC, BIDS, ASKS)]  # 10초 공백 뒤 호가
    lg = League(["claude"], 0)
    for i, e in enumerate(evs):
        if i == 3:
            lg.submit("claude", [mkt()])  # 공백 직전에 낸 주문 → 공백 뒤 스냅샷에서 체결
        lg.step(e)
    fills = lg.engines["claude"].fills
    assert fills and all(f["fill_grade"] != "A" for f in fills)

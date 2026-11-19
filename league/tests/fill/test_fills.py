"""명세 13절: 시장가 VWAP·depth 벌점·자기 충격·지정가 관통/대기열·ALO·지연·등급 경계."""
from decimal import Decimal as D

from conftest import book, fills_of, lim, mkt, run, trade

from league.fill import ClockEvent
from league.fill.market import MarketState

ASKS = [(100, 1), (101, 2), (102, 5)]
BIDS = [(99, 1), (98, 2), (97, 5)]


def test_market_walks_levels_vwap_and_fee():
    ev = [book(0, BIDS, ASKS), book(1000, BIDS, ASKS)]
    eng, res = run(ev, [(0, mkt("buy", "2.5"))])
    assert fills_of(res) == [(D(100), D(1), "taker", "B"), (D(101), D("1.5"), "taker", "B")]
    vwap = (D(100) * 1 + D(101) * D("1.5")) / D("2.5")
    assert vwap == D("100.6")
    fee = sum(D(f["fee"]) for f in res.fills)
    assert fee == (D(100) + D("151.5")) * D("0.00045")
    assert res.orders[0]["state"] == "filled"
    assert eng.account.qty("BTC-USDC-PERP") == D("2.5")


def test_latency_uses_first_snapshot_after_arrival_no_future():
    ev = [book(1000, BIDS, [(100, 10)]), book(1200, BIDS, [(105, 10)]),
          book(1300, BIDS, [(110, 10)]), book(1400, BIDS, [(120, 10)])]
    _, res = run(ev, [(1000, mkt("buy", 1))])          # 도착 1250 → 1300 스냅샷
    assert fills_of(res)[0][0] == D(110)
    _, res0 = run(ev, [(1000, mkt("buy", 1))], latency_ms=0)  # 지연 0 → 같은 시각 스냅샷은 이미 지나감, 다음 것
    assert fills_of(res0)[0][0] == D(105)


def test_depth_exhausted_penalty():
    ev = [book(0, BIDS, [(100, 1), (101, 1)]), book(1000, BIDS, [(100, 1), (101, 1)])]
    _, res = run(ev, [(0, mkt("buy", 3))])
    got = fills_of(res)
    assert [g[0] for g in got] == [D(100), D(101), D(101) * D("1.001")]
    assert res.meta["flags"]["depth_exhausted"] == 1


def test_max_slippage_cancels_rest():
    ev = [book(0, BIDS, [(100, 1), (110, 5)]), book(1000, BIDS, [(100, 1), (110, 5)])]
    _, res = run(ev, [(0, mkt("buy", 3))])
    assert fills_of(res) == [(D(100), D(1), "taker", "B")]
    assert res.orders[0]["state"] == "canceled"


def test_self_impact_within_snapshot_and_reset():
    ev = [book(0, BIDS, ASKS), book(1000, BIDS, ASKS), ClockEvent(1500), book(2000, BIDS, ASKS)]
    _, res = run(ev, [(0, mkt("buy", "1.5")), (0, mkt("buy", 1)), (1500, mkt("buy", 1))])
    assert fills_of(res, 0) == [(D(100), D(1), "taker", "B"), (D(101), D("0.5"), "taker", "B")]
    assert fills_of(res, 1) == [(D(101), D(1), "taker", "B")]   # 100 단계는 이미 먹음
    assert fills_of(res, 2)[0][0] == D(100)                       # 새 스냅샷 → 회복


def _a_book(ts):
    return book(ts, [(99, 5), (98, 5)], [(100, 1), (101, 5)])


def test_limit_queue_same_price_grade_a():
    ev = [_a_book(0), trade(500, 100, "0.1", "buy"), _a_book(1000), _a_book(2000),
          trade(2100, 99, 3, "sell"), trade(2150, 99, 3, "buy"),   # 같은 쪽 체결은 대기열과 무관
          trade(2200, 99, 4, "sell"), _a_book(3000)]
    _, res = run(ev, [(1000, lim("buy", 2, 99))])
    # 대기열 5 → 3 소진(2 남음) → 4 소진(−2) → 초과 2 체결
    assert fills_of(res) == [(D(99), D(2), "maker", "A")]
    assert res.fills[0]["ts_ms"] == 2200


def test_limit_through_fills_all_grade_a():
    ev = [_a_book(0), trade(500, 100, "0.1", "buy"), _a_book(1000), _a_book(2000),
          trade(2100, D("98.5"), "0.01", "sell"), _a_book(3000)]
    _, res = run(ev, [(1000, lim("buy", 2, 99))])
    assert fills_of(res) == [(D(99), D(2), "maker", "A")]


def test_limit_grade_b_needs_through_not_touch():
    ev = [book(0, BIDS, ASKS), book(1000, BIDS, ASKS),
          book(2000, BIDS, [(D("99"), 1)]),        # 닿기만 → 체결 안 됨
          book(3000, BIDS, [(D("98.9"), 1)])]      # 관통 → 지정가로 전부
    _, res = run(ev, [(0, lim("buy", 2, 99))])
    assert fills_of(res) == [(D(99), D(2), "maker", "B")]
    assert res.fills[0]["ts_ms"] == 3000


def test_marketable_limit_takes_up_to_limit_then_rests():
    ev = [book(0, BIDS, ASKS), book(1000, BIDS, ASKS), book(2000, BIDS, [(D("100.5"), 9)])]
    _, res = run(ev, [(0, lim("buy", 3, 101))])
    assert fills_of(res) == [(D(100), D(1), "taker", "B"), (D(101), D(2), "taker", "B")]
    _, res = run(ev, [(0, lim("buy", 4, "100.9"))])   # 100까지만 테이커, 나머지 대기 → 100.5 관통 체결
    assert fills_of(res) == [(D(100), D(1), "taker", "B"), (D("100.9"), D(3), "maker", "B")]


def test_alo_rejected_when_marketable():
    ev = [book(0, BIDS, ASKS), book(1000, BIDS, ASKS)]
    _, res = run(ev, [(0, lim("buy", 1, 100, tif="ALO")), (0, lim("buy", 1, 99, tif="ALO"))])
    assert res.orders[0]["state"] == "rejected" and res.fills == []
    assert res.orders[1]["state"] == "acknowledged"


def test_ioc_remainder_canceled():
    ev = [book(0, BIDS, ASKS), book(1000, BIDS, ASKS)]
    _, res = run(ev, [(0, lim("buy", 5, 101, tif="IOC"))])
    assert sum(q for _, q, _, _ in fills_of(res)) == D(3)
    assert res.orders[0]["state"] == "canceled"


def test_grade_boundary_snapshot_interval():
    st = MarketState("X")
    st.apply(trade(0, 100, 1, "buy"))
    st.apply(book(1000, BIDS, ASKS))
    assert st.grade(1000) == "B"              # 간격을 아직 모름 → 보수적으로 B
    st.apply(book(3000, BIDS, ASKS))
    assert st.grade(3000) == "A"              # 간격 2000 = 경계 포함
    assert st.grade(5001) == "B"              # 나이 2001
    st.apply(book(5001, BIDS, ASKS))
    assert st.grade(5001) == "B"              # 간격 2001
    nt = MarketState("Y")
    nt.apply(book(0, BIDS, ASKS)); nt.apply(book(1000, BIDS, ASKS))
    assert nt.grade(1000) == "B"              # 체결 데이터 없음


def test_reduce_only_caps_and_cancels():
    ev = [book(0, BIDS, ASKS), book(1000, BIDS, ASKS), ClockEvent(1500), book(2000, BIDS, ASKS),
          ClockEvent(2500), book(3000, BIDS, ASKS)]
    _, res = run(ev, [(0, mkt("sell", 1, reduce_only=True)),
                      (1500, mkt("buy", 1)), (2500, mkt("sell", 3, reduce_only=True))])
    assert res.orders[0]["state"] == "canceled" and fills_of(res, 0) == []
    assert sum(q for _, q, _, _ in fills_of(res, 2)) == D(1)
    assert res.metrics["open_positions"] == []


def test_cancel_after_latency():
    from league.fill import Cancel
    ev = [book(0, BIDS, ASKS), book(1000, BIDS, ASKS), book(2000, BIDS, [(98, 1)]), ClockEvent(1100)]
    _, res = run(ev, [(0, lim("buy", 1, 99, tag="a")), (1000, Cancel(tag="a"))])
    assert res.orders[0]["state"] == "canceled" and res.fills == []

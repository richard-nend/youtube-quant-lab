"""명세 13절: 스탑 트리거(마크/최종가) · 등급 C(캔들)."""
from decimal import Decimal as D

from conftest import BN_BTC, HL_BTC, book, fills_of, lim, mkt, run, trade

from league.fill import CandleEvent, ClockEvent, CtxEvent, OrderRequest, Trigger

BIDS = [(94, 10)]
ASKS = [(95, 10)]


def _stop_sell(iid, by=None):
    return OrderRequest(iid, "sell", "stop_market", 1, trigger=Trigger(95, by), reduce_only=False)


def test_hl_stop_triggers_on_mark_not_last():
    ev = [book(0, BIDS, ASKS), CtxEvent(1000, HL_BTC, mark=96),
          trade(2000, 94, 1, "sell"),                 # 최종가가 넘어가도 마크 기준이면 발동 안 함
          book(2500, BIDS, ASKS),
          CtxEvent(3000, HL_BTC, mark="94.9"), book(3000, BIDS, ASKS)]
    _, res = run(ev, [(0, _stop_sell(HL_BTC))])
    assert res.orders[0]["trigger"]["by"] == "mark"
    assert len(res.fills) == 1 and res.fills[0]["ts_ms"] == 3000
    assert res.meta["flags"]["trigger_mark"] == 1


def test_binance_stop_triggers_on_last_by_default():
    bk = lambda ts: book(ts, BIDS, ASKS, iid=BN_BTC)  # noqa: E731
    ev = [bk(0), CtxEvent(1000, BN_BTC, mark=90),        # 마크는 무시
          bk(1500), trade(2000, "94.5", 1, "sell", iid=BN_BTC), bk(2100)]
    _, res = run(ev, [(0, _stop_sell(BN_BTC))])
    assert res.orders[0]["trigger"]["by"] == "last"
    assert [f["ts_ms"] for f in res.fills] == [2100]


def test_take_profit_direction_and_explicit_by():
    ev = [book(0, BIDS, ASKS), trade(1000, 104, 1, "buy"), trade(2000, 106, 1, "buy"), book(2100, BIDS, ASKS)]
    tp = OrderRequest(HL_BTC, "sell", "take_profit_market", 1, trigger=Trigger(105, "last"))
    _, res = run(ev, [(0, tp)])
    assert [f["ts_ms"] for f in res.fills] == [2100]


def _bar(ts, o, h, l, c, v=100):
    return CandleEvent(ts, HL_BTC, 60_000, o, h, l, c, v)


def test_grade_c_market_next_bar_open_plus_slippage():
    ev = [ClockEvent(0), _bar(0, 90, 91, 89, 90), _bar(60_000, 100, 101, 99, 100), _bar(120_000, 200, 200, 200, 200)]
    _, res = run(ev, [(0, mkt("buy", 1))])
    # 도착 250ms → 시작이 도착 이후인 첫 봉(60000)의 시가. 슬리피지 = 2bp + 1bp × (1/100 거래량 = 1%)
    assert fills_of(res) == [(D(100) * D("1.0003"), D(1), "taker", "C")]
    assert res.meta["flags"]["candle_estimate"] == 1
    assert res.meta["grades"]["by_count_pct"]["C"] == 100.0



def test_grade_c_zero_volume_index_pays_half_spread_only():
    """11-12 연구 요청: 거래량 0 봉은 100%(102bp)로 보지만, 지수(현물 다리 대용 -INDEX)는 거래량 자료가 없을 뿐이라 반 스프레드만."""
    from league.fill.fillmodels import FillModelC, FillParams
    m = FillModelC(FillParams())
    assert m.slippage(D(1), D(0), "BTC-USD-INDEX") == D("0.0002")
    assert m.slippage(D(1), D(0), "BTC-USDT-PERP") == min(D("0.0102"), FillParams().max_slippage)
    assert m.slippage(D(1), D(100), "BTC-USD-INDEX") == D("0.0003")

def test_grade_c_limit_needs_through():
    ev = [ClockEvent(0), _bar(60_000, 100, 101, 99, 100), _bar(120_000, 100, 101, D("98.9"), 100)]
    _, res = run(ev, [(0, lim("buy", 1, 99))])   # 첫 봉 저가 99 = 닿기만 → 다음 봉 98.9 관통
    assert fills_of(res) == [(D(99), D(1), "maker", "C")]
    assert res.fills[0]["ts_ms"] == 120_000 + 60_000   # 봉이 끝나야 알 수 있다


def test_grade_c_alo_rejected_vs_open():
    ev = [ClockEvent(0), _bar(60_000, 100, 101, 99, 100)]
    _, res = run(ev, [(0, lim("buy", 1, 100, tif="ALO"))])
    assert res.orders[0]["state"] == "rejected"


def test_daily_equity_sample_at_midnight_even_with_intraday_fills():
    """11-12 연구 요청 b: 낮 체결이 표본 시계를 되돌려도 자정 경계를 넘는 첫 이벤트에서 자산을 기록(그날 마지막 체결 시각 값이 아님)."""
    H, DAY = 3_600_000, 86_400_000
    ev = [ClockEvent(0)] + [_bar(h * H, 100 + h, 100 + h, 100 + h, 100 + h) for h in range(0, 50)]
    eng, res = run(ev, [(5 * H, mkt("buy", 1)), (29 * H, mkt("buy", 1))], equity_sample_ms=DAY)
    ts = [t for t, _ in res.equity_curve]
    assert any(DAY <= t < DAY + H for t in ts)   # 자정 직후 첫 이벤트에서 표본

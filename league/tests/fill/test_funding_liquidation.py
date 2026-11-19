"""명세 13절: HL 매시간·바이낸스 8시간 펀딩 정산과 방향 · 청산가 공식 · 부분·백스톱 청산."""
from decimal import Decimal as D

from conftest import BN_BTC, HL_BTC, book, cfg, mkt, run

from league.fill import ClockEvent, CtxEvent, Engine, FundingEvent, ScriptedStrategy
from league.fill.margin import binance_liquidation_price, hl_liquidation_price

H = 3_600_000


def _funding_rows(res):
    return [(e["ts_ms"], D(e["amount"])) for e in res.ledger if e["type"] == "funding"]


def test_hl_hourly_funding_long_pays_oracle_price():
    ev = [book(0, [(99_990, 5)], [(100_000, 5)]), book(1000, [(99_990, 5)], [(100_000, 5)])]
    for h in range(1, 4):
        ev.append(CtxEvent(h * H - 1000, HL_BTC, mark=100_100, index=100_000, funding_rate="0.0001"))
    ev.append(FundingEvent(2 * H, HL_BTC, "0.0002"))   # 확정 기록이 있으면 그것을 쓴다
    ev.append(ClockEvent(3 * H + 1))
    _, res = run(ev, [(0, mkt("buy", 1))])
    assert _funding_rows(res) == [(H, D("-10")), (2 * H, D("-20")), (3 * H, D("-10"))]
    assert "funding_imputed" not in res.meta["flags"]


def test_hl_short_receives():
    ev = [book(0, [(100_000, 5)], [(100_010, 5)]), book(1000, [(100_000, 5)], [(100_010, 5)]),
          CtxEvent(H - 10, HL_BTC, mark=100_000, index=100_000, funding_rate="0.0001"), ClockEvent(H + 1)]
    _, res = run(ev, [(0, mkt("sell", 2))])
    assert _funding_rows(res) == [(H, D("20"))]


def test_binance_8h_funding_mark_price_short_receives():
    bk = lambda ts: book(ts, [(100_000, 5)], [(100_010, 5)], iid=BN_BTC)  # noqa: E731
    ev = [CtxEvent(0, BN_BTC, mark=100_000, funding_rate="0.0001", next_funding_ms=8 * H), bk(0), bk(1000)]
    for h in range(1, 10):
        ev.append(CtxEvent(h * H - 1000, BN_BTC, mark=100_050, index=99_000, funding_rate="0.0001",
                           next_funding_ms=8 * H if h <= 8 else 16 * H))
        ev.append(ClockEvent(h * H + 1))
    _, res = run(ev, [(0, mkt("sell", 1, iid=BN_BTC))])
    assert _funding_rows(res) == [(8 * H, D("10.005"))]   # 1 × 마크 100050 × 0.0001, 8시간 한 번만


def test_funding_imputed_when_ctx_stale():
    ev = [book(0, [(99_990, 5)], [(100_000, 5)]), book(1000, [(99_990, 5)], [(100_000, 5)]),
          CtxEvent(2000, HL_BTC, mark=100_000, index=100_000, funding_rate="0.0001"), ClockEvent(H + 1)]
    _, res = run(ev, [(0, mkt("buy", 1))])
    assert _funding_rows(res) == [(H, D("-10"))]
    assert res.meta["flags"]["funding_imputed"] == 1


def test_hl_liquidation_price_formula_consistent():
    # 격리 롱 1 BTC @100,000, 증거금 10,000, 최대 레버리지 40 → 유지 증거금률 1/80
    l = D(1) / D(80)
    margin_available = D(10_000) - D(100_000) * l
    liq = hl_liquidation_price(D(100_000), D(1), 1, margin_available, l)
    assert abs(liq - D("91139.24050632911392405063291")) < D("1e-9")
    equity = D(10_000) + (liq - D(100_000))           # 청산가에서 자산 = 유지 증거금
    assert abs(equity - liq * l) < D("1e-9")
    short = hl_liquidation_price(D(100_000), D(1), -1, margin_available, l)
    assert abs((D(10_000) - (short - D(100_000))) - short * l) < D("1e-9")


def test_binance_liquidation_price_isolated_long_matches_closed_form():
    ep, lev, mmr = D(100_000), D(10), D("0.004")
    lp = binance_liquidation_price(wallet=ep / lev, pos=D(1), entry=ep, side=1, mmr=mmr)
    assert abs(lp - ep * (1 - 1 / lev) / (1 - mmr)) < D("1e-9")


def _liq_events(marks, iid=HL_BTC, start=1000):
    ev = [book(0, [(10_000, 100)], [(10_000, 100)], iid=iid), book(500, [(10_000, 100)], [(10_000, 100)], iid=iid)]
    for i, m in enumerate(marks):
        ts = start + i * 10_000
        ev += [CtxEvent(ts, iid, mark=m, index=m), book(ts, [(m, 100)], [(m + 1, 100)], iid=iid)]
    return ev


def test_hl_liquidation_full_when_small():
    # 현금 1,000, 롱 1 @10,000. 유지 증거금 = 마크 × 1.25%. 9,100에서 자산 100 < 113.75, 2/3(75.8)보다는 큼 → 전량 시장가
    eng = Engine(cfg(initial_cash=1000))
    res = eng.run(_liq_events([9_500, 9_100]), ScriptedStrategy([(0, mkt("buy", 1))]))
    liq = [e for e in res.ledger if e["type"] == "liquidation"]
    assert len(liq) == 1 and eng.account.qty(HL_BTC) == 0
    assert res.meta["flags"]["liquidation"] == 1 and "liquidation_backstop" not in res.meta["flags"]


def test_hl_partial_liquidation_20pct_with_30s_wait():
    # 롱 20 @10,000 (명목 200k > 100k). 9,600: 자산 2,000 < 유지 2,400, > 2/3(1,600) → 20%씩
    eng = Engine(cfg(initial_cash=10_000))
    marks = [9_600, 9_600, 9_600, 9_600]   # 10초 간격 → 30초 뒤 두 번째 부분 청산
    res = eng.run(_liq_events(marks), ScriptedStrategy([(0, mkt("buy", 20))]))
    sells = [D(f["qty"]) for f in res.fills if f["ts_ms"] >= 1000]
    assert sells[0] == D(4)                  # 20 × 20%
    assert res.meta["flags"]["liquidation_partial"] >= 2
    liq_ts = sorted({e["ts_ms"] for e in res.ledger if e["type"] == "liquidation"})
    assert liq_ts[1] - liq_ts[0] >= 30_000


def test_hl_backstop_zeroes_cross_equity():
    eng = Engine(cfg(initial_cash=10_000))
    res = eng.run(_liq_events([9_550]), ScriptedStrategy([(0, mkt("buy", 20))]))
    assert res.meta["flags"]["liquidation_backstop"] == 1
    assert eng.account.qty(HL_BTC) == 0 and D(res.metrics["final_equity"]) == 0
    assert any(e["ref_id"] == "backstop" for e in res.ledger)


def test_binance_liquidation_closes_all_with_estimated_mmr_flag():
    from league.fill import EngineConfig, InstrumentSpec
    c = EngineConfig(instruments={BN_BTC: InstrumentSpec(BN_BTC, "binance_usdm", quote="USDT")}, initial_cash=1000)
    eng = Engine(c)
    res = eng.run(_liq_events([9_500, 9_020], iid=BN_BTC), ScriptedStrategy([(0, mkt("buy", 1, iid=BN_BTC))]))
    assert eng.account.qty(BN_BTC) == 0
    assert res.meta["assumptions"]["mmr_estimated"] == [BN_BTC]


def _candles(rows, iid=BN_BTC, step=86_400_000):
    from league.fill import CandleEvent
    return [CandleEvent(k * step, iid, step, o, h, l, c, 1e12) for k, (o, h, l, c) in enumerate(rows)]


def _bn_engine(cash):
    from league.fill import EngineConfig, InstrumentSpec
    return Engine(EngineConfig(instruments={BN_BTC: InstrumentSpec(BN_BTC, "binance_usdm", mmr="0.004", quote="USDT")}, initial_cash=cash))


def test_candle_only_short_liquidated_inside_the_spike_bar():
    """#61: 캔들만 있는 구간 — 급등 봉의 고가에서 자산 < 유지 증거금이면 그 봉 안 청산가에서 청산(종가가 돌아와도)."""
    eng = _bn_engine(1000)
    flat = (10_000, 10_000, 10_000, 10_000)
    res = eng.run(_candles([flat, flat, (10_000, 11_500, 9_900, 10_000), flat]),
                  ScriptedStrategy([(0, mkt("sell", 1, iid=BN_BTC))]))
    assert eng.account.qty(BN_BTC) == 0
    assert res.meta["flags"]["liquidation_intrabar"] == 1
    liq = [f for f in res.fills if D(f["qty"]) == 1 and D(f["price"]) > 10_000]
    p = D(liq[0]["price"])
    entry = D(next(f["price"] for f in res.fills))
    # 청산가: 현금 − (p − 진입) ≈ 유지 증거금(p × 0.004) — 이분법 오차 안
    assert abs((1000 - D(next(f["fee"] for f in res.fills)) - (p - entry)) - p * D("0.004")) < D("0.01")
    assert 10_000 < p < 11_500
    assert D(res.metrics["final_equity"]) >= 0


def test_candle_only_gap_open_liquidates_at_open():
    eng = _bn_engine(1000)
    flat = (10_000, 10_000, 10_000, 10_000)
    res = eng.run(_candles([flat, flat, flat, (12_000, 12_500, 11_900, 12_200)]), ScriptedStrategy([(0, mkt("sell", 1, iid=BN_BTC))]))
    liq = [f for f in res.fills if D(f["price"]) >= 12_000]
    assert D(liq[0]["price"]) == 12_000 and eng.account.qty(BN_BTC) == 0
    assert D(res.metrics["final_equity"]) == 0          # 손실 상한 = 자산(보험 기금)


def test_candle_only_no_liquidation_when_extreme_is_safe():
    eng = _bn_engine(1000)
    flat = (10_000, 10_000, 10_000, 10_000)
    res = eng.run(_candles([flat, flat, flat, (10_000, 10_500, 9_900, 10_100)]), ScriptedStrategy([(0, mkt("sell", 1, iid=BN_BTC))]))
    assert eng.account.qty(BN_BTC) == -1 and "liquidation" not in res.meta["flags"]


def test_delist_settles_open_position_at_price_and_rejects_later_orders():
    """#62: 상장폐지 사건 — 열린 포지션을 정산가로 닫고(수수료 없음, ref delisted), 대기 주문 취소, 이후 주문 거절."""
    from league.fill import DelistEvent
    eng = _bn_engine(10_000)
    flat = (10_000, 10_000, 10_000, 10_000)
    ev = _candles([flat, flat, (10_000, 10_500, 9_900, 10_400)]) + [DelistEvent(3 * 86_400_000, BN_BTC, "10400"), ClockEvent(3 * 86_400_000 + 10)]
    res = eng.run(ev, ScriptedStrategy([(0, mkt("buy", 1, iid=BN_BTC)), (3 * 86_400_000 + 1, mkt("buy", 1, iid=BN_BTC))]))
    assert eng.account.qty(BN_BTC) == 0
    row = next(e for e in res.ledger if e["ref_id"] == f"delisted:{BN_BTC}")
    entry = D(res.fills[0]["price"])
    assert D(row["amount"]) == D("10400") - entry
    assert res.meta["flags"]["delisted_settled"] == 1 and res.meta["flags"]["order_rejected_delisted"] == 1

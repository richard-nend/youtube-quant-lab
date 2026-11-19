"""현물 상품(market_type=SPOT): 현물 수수료 표, 펀딩 없음, 유지 증거금 0 — 캐리(현물 롱 + 무기한 숏)."""
from decimal import Decimal as D

from conftest import HL_BTC, book, mkt

from league.fill import CtxEvent, Engine, EngineConfig, FundingEvent, InstrumentSpec, ScriptedStrategy
from league.fill.events import ClockEvent

H = 3_600_000
SPOT = "UBTC-USDC-SPOT"


def _cfg():
    return EngineConfig(instruments={
        HL_BTC: InstrumentSpec(HL_BTC, "hyperliquid", max_leverage=40),
        SPOT: InstrumentSpec(SPOT, "hyperliquid", market_type="SPOT")})


def test_spot_fee_no_funding_no_maintenance():
    ev = []
    for ts in (0, 1000):
        ev += [book(ts, [(99_990, 5)], [(100_000, 5)]), book(ts, [(99_990, 5)], [(100_000, 5)], iid=SPOT)]
    ev += [CtxEvent(H - 10, HL_BTC, mark=100_000, index=100_000, funding_rate="0.0001"),
           CtxEvent(H - 10, SPOT, mark=100_000, funding_rate="0.0001"),   # 현물에 펀딩 값이 와도 정산 안 함
           FundingEvent(H, HL_BTC, "0.0001"), ClockEvent(H + 1)]
    eng = Engine(_cfg())
    res = eng.run(ev, ScriptedStrategy([(0, mkt("buy", 1, iid=SPOT)), (0, mkt("sell", 1))]))
    fees = {e["instrument_id"]: D(e["amount"]) for e in res.ledger if e["type"] == "fee"}
    assert fees[SPOT] == D("-70")          # 100,000 × 0.070%
    assert fees[HL_BTC] == D("-44.9955")   # 99,990 × 0.045%
    funding = [(e["instrument_id"], D(e["amount"])) for e in res.ledger if e["type"] == "funding"]
    assert funding == [(HL_BTC, D("10"))]  # 숏만 받는다
    assert eng.spec(SPOT).maintenance(D(1), D(100_000)) == 0
    assert res.meta["assumptions"]["fees"]["hyperliquid:SPOT"] == {"taker": "0.0007", "maker": "0.0004"}
    assert res.meta["assumptions"]["mmr_estimated"] == []


def test_future_has_no_funding_and_settled_sampling():
    """#54: 분기물(market_type=FUTURE)은 펀딩 정산 없음. sample_settled 면 같은 시각 봉이 다 반영된 뒤에만 자산을 기록한다."""
    from league.fill import CandleEvent
    fut = "BTC-USD-250328-FUT"
    ix = "BTC-USD-INDEX"
    specs = {fut: InstrumentSpec(fut, "binance_coinm", market_type="FUTURE"),
             ix: InstrumentSpec(ix, "binance_coinm", market_type="SPOT")}

    def events():
        ev = []
        for k, px in enumerate([100.0, 100.0, 100.0, 120.0, 120.0]):   # 체결(둘째 봉 시가) 뒤 넷째 봉에서 둘 다 20% 상승
            for iid in (fut, ix):
                ev.append(CandleEvent(k * H, iid, H, px, px, px, px, 1e12))
        ev.append(FundingEvent(2 * H, fut, "0.01"))               # 와도 정산 안 함
        return ev

    def curve(settled: bool):
        eng = Engine(EngineConfig(instruments=specs, initial_cash=D(1000), equity_sample_ms=1, sample_settled=settled))
        res = eng.run(events(), ScriptedStrategy([(0, mkt("buy", 1, iid=ix)), (0, mkt("sell", 1, iid=fut))]))
        assert not [e for e in res.ledger if e["type"] == "funding"]
        return res, [D(v) for _, v in res.equity_curve]

    res, settled = curve(True)
    _, old = curve(False)
    assert max(settled) - min(settled) < D(1)                     # 헤지 → 수수료·체결 미끄러짐만큼만 움직임
    assert max(old) - min(old) >= D(19)                           # 종전: 한쪽만 오른 순간이 찍힘


def test_negative_balance_after_all_positions_closed_is_covered():
    """#56: 청산 뒤 포지션이 없는데 잔고가 음수면 보험 기금이 메워 0(교차 증거금 손실 상한 = 자산)."""
    from league.fill import ClockEvent
    specs = {"BTC-USDT-PERP": InstrumentSpec("BTC-USDT-PERP", "binance_usdm")}
    eng = Engine(EngineConfig(instruments=specs, initial_cash=D(1000)))
    eng.account.cash = D(-250)
    eng.step(ClockEvent(1_000))
    assert eng.account.cash == 0
    assert eng.ledger[-1]["type"] == "liquidation" and eng.ledger[-1]["ref_id"] == "insurance_fund"
    assert D(eng.ledger[-1]["amount"]) == D(250)

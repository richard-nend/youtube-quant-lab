"""명세 13절 결과 재현성 + 10절 메타 + 출력 행이 schemas/ 정본(order·fill·ledger_entry)을 통과하는지."""
import json
from decimal import Decimal as D
from pathlib import Path

from conftest import BN_BTC, HL_BTC, book, cfg, lim, mkt, trade
from jsonschema import Draft202012Validator
from referencing import Registry, Resource

from league.fill import ENGINE_VERSION, ClockEvent, CtxEvent, Engine, FundingEvent, OrderRequest, ScriptedStrategy, Trigger

SCHEMAS = Path(__file__).resolve().parent / "schemas"
H = 3_600_000


def _scenario():
    ev, steps = [], []
    for i in range(0, 3 * H, 1000 * 60):
        p = 100_000 + (i // 60_000) % 7 * 10
        ev += [book(i, [(p - 5, 2), (p - 10, 5)], [(p, 2), (p + 5, 5)]), trade(i + 10, p, "0.3", "buy"),
               book(i + 1000, [(p - 5, 2)], [(p, 2)]),
               CtxEvent(i + 500, HL_BTC, mark=p, index=p - 2, funding_rate="0.00002")]
        ev += [book(i, [(p - 5, 2)], [(p, 2)], iid=BN_BTC),
               CtxEvent(i + 500, BN_BTC, mark=p, funding_rate="0.0001", next_funding_ms=8 * H)]
    ev.append(FundingEvent(H, HL_BTC, "0.00003"))
    ev.append(ClockEvent(3 * H + 5))
    steps = [(0, mkt("buy", "1.5", builder_fee_bps="1")), (60_000, lim("sell", 1, 100_030, tag="tp")),
             (120_000, OrderRequest(HL_BTC, "sell", "stop_market", "0.2", trigger=Trigger(100_005))),
             (180_000, mkt("sell", 1, iid=BN_BTC)), (240_000, lim("buy", 1, 99_000, tif="ALO", iid=BN_BTC))]
    return ev, steps


def _run():
    ev, steps = _scenario()
    eng = Engine(cfg(sources=[{"venue": "hyperliquid", "grade": "green", "internal_only": False},
                              {"venue": "binance_usdm", "grade": "red", "internal_only": True}]))
    return eng.run(ev, ScriptedStrategy(steps))


def test_same_input_same_output():
    a, b = _run().to_dict(), _run().to_dict()
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    assert len(a["fills"]) >= 4 and any(e["type"] == "funding" for e in a["ledger"])


def test_meta_and_metrics():
    r = _run()
    m = r.meta
    assert m["engine_version"] == ENGINE_VERSION
    assert set(m["grades"]["by_count_pct"]) == {"A", "B", "C"}
    assert abs(sum(m["grades"]["by_notional_pct"].values()) - 100) < 0.5
    assert m["internal_only_used"] is True and m["assumptions"]["latency_ms"] == 250
    assert m["data"]["instruments"] == [HL_BTC, BN_BTC]
    # 원장 합 = 현금 변화
    total = sum(D(e["amount"]) for e in r.ledger)
    assert D(r.metrics["final_cash"]) - D(r.metrics["initial_cash"]) == total
    assert any(e["type"] == "builder_fee" for e in r.ledger)


def _validator(name: str) -> Draft202012Validator:
    reg = Registry()
    for p in SCHEMAS.glob("**/*.json"):
        if "node_modules" in p.parts or "generated" in p.parts or "examples" in p.parts:
            continue
        doc = json.loads(p.read_text())
        if "$id" in doc:
            reg = reg.with_resource(doc["$id"], Resource.from_contents(doc))
    schema = json.loads((SCHEMAS / "trading" / f"{name}.json").read_text())
    return Draft202012Validator(schema, registry=reg)


def test_rows_match_trading_schemas():
    r = _run()
    for name, rows in (("order", r.orders), ("fill", r.fills), ("ledger_entry", r.ledger)):
        v = _validator(name)
        for row in rows:
            errs = list(v.iter_errors(row))
            assert not errs, (name, row, [e.message for e in errs])

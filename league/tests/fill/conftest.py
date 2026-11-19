from decimal import Decimal

import pytest

from league.fill import (BookEvent, ClockEvent, CtxEvent, Engine, EngineConfig, InstrumentSpec,
                              OrderRequest, ScriptedStrategy, TradeEvent)

HL_BTC = "BTC-USDC-PERP"
BN_BTC = "BTC-USDT-PERP"


def specs():
    return {HL_BTC: InstrumentSpec(HL_BTC, "hyperliquid", max_leverage=40),
            BN_BTC: InstrumentSpec(BN_BTC, "binance_usdm", mmr="0.004", quote="USDT")}


def cfg(**kw) -> EngineConfig:
    return EngineConfig(instruments=specs(), **kw)


def book(ts, bids, asks, iid=HL_BTC):
    return BookEvent(ts, iid, bids, asks)


def trade(ts, price, size, side, iid=HL_BTC):
    return TradeEvent(ts, iid, price, size, side)


def run(events, steps, **kw):
    eng = Engine(cfg(**kw))
    res = eng.run(events, ScriptedStrategy(steps))
    return eng, res


def mkt(side, qty, iid=HL_BTC, **kw):
    return OrderRequest(iid, side, "market", qty, **kw)


def lim(side, qty, price, iid=HL_BTC, **kw):
    return OrderRequest(iid, side, "limit", qty, price=price, **kw)


def fills_of(res, i=0):
    oid = res.orders[i]["ord_id"]
    return [(Decimal(f["price"]), Decimal(f["qty"]), f["liquidity"], f["fill_grade"])
            for f in res.fills if f["ord_id"] == oid]


@pytest.fixture
def D():
    return Decimal

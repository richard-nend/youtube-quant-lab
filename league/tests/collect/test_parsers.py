import json
from pathlib import Path

import pytest

from league.collect import toobit as tbm
from league.collect.base import CollectorSpec
from league.collect.binance import BinanceUsdmCollector
from league.collect.util import toobit_swap

FX = Path(__file__).parent / "fixtures"


def load(name):
    return json.loads((FX / f"{name}.json").read_text())


def bn(symbols=("BTCUSDT", "ETHUSDT")):
    out = []
    return BinanceUsdmCollector(CollectorSpec("binance_usdm", tuple(symbols)), out.append), out


def toobit():
    c = tbm.ToobitUsdtmCollector(CollectorSpec("toobit_usdtm", ("BTCUSDT",)), lambda r: None)
    c.mult = {"BTC-SWAP-USDT": 0.001}
    return c


def test_bn_aggtrade_maker_flag():
    c, out = bn()
    c.emit(c.handle(load("bn_aggtrade")))
    t = out[0]
    assert t.side == "sell" and t.price == 67260.10 and t.trade_id == "555000111"


def test_bn_depth():
    c, out = bn()
    c.emit(c.handle(load("bn_depth")))
    assert out[0].bids[0] == [2510.05, 10.0] and out[0].ts_ms == 1761400001998


def test_bn_ws_urls_split_market_and_public():
    c, _ = bn(("BTCUSDT",))
    market, public = c.ws_urls()
    assert "/market/stream" in market and "/public/stream" in public
    assert "btcusdt@aggTrade" in market and "btcusdt@depth20@100ms" in public and "depth" not in market


def test_toobit_symbol_converted_from_concat_form():
    assert toobit().syms == ["BTC-SWAP-USDT"] and toobit_swap("SOL-SWAP-USDT") == "SOL-USDT-PERP"


def test_toobit_trades_converted_to_coin():
    raw = load("tb_ws_trade")
    tr = toobit().handle(raw)
    assert len(tr) == len(raw["data"])
    assert tr[0].size == pytest.approx(float(raw["data"][0]["q"]) * 0.001)
    assert tr[0].side == ("sell" if raw["data"][0]["m"] else "buy")
    assert tr[0].instrument_id == "BTC-USDT-PERP" and tr[0].trade_id == raw["data"][0]["v"]


def test_toobit_depth_and_ignored_messages():
    c = toobit()
    raw = load("tb_ws_depth")
    b = c.handle(raw)[0]
    assert b.bids[0][1] == pytest.approx(float(raw["data"][0]["b"][0][1]) * 0.001)
    assert c.handle(load("tb_ws_pong")) == []
    assert c.handle({"code": "-10004", "desc": "Invalid topic!"}) == []


def test_toobit_ctx():
    m = tbm.parse_ctx("BTC-SWAP-USDT", load("tb_mark"), load("tb_funding")[0], load("tb_oi"), load("tb_index"), 5)
    assert m.funding_interval_h == 8.0 and m.mark_price and m.index_price and m.open_interest
    assert m.next_funding_ms and m.funding_rate is not None


def test_toobit_contracts():
    tb = tbm.parse_contracts(load("tb_exchange_info"), 1)[0]
    assert tb.contract_multiplier == 0.001 and tb.max_leverage == 200 and tb.asset == "BTC"

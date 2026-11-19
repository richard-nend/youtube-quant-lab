from decimal import Decimal

from league.collect.models import BookSnapshot, Candle, Funding, MarketCtx, Trade
from league.events import INSTRUMENT_OF, to_engine_event
from league.fill import BookEvent, CtxEvent, FundingEvent, TradeEvent
from league.venues import BTC, ETH, TOOBIT


def test_instrument_of():
    assert INSTRUMENT_OF["BTC-SWAP-USDT"] == BTC and INSTRUMENT_OF["ETH-SWAP-USDT"] == ETH


def test_book():
    r = BookSnapshot(1000, TOOBIT, "BTC-SWAP-USDT", [[100.0, 0.5], [99.0, 1.0]], [[101.0, 0.2], [102.0, 3.0]], 1001)
    e = to_engine_event(r)
    assert isinstance(e, BookEvent) and e.instrument_id == BTC
    assert [p for p, _ in e.bids] == [Decimal("100.0"), Decimal("99.0")]
    assert [p for p, _ in e.asks] == [Decimal("101.0"), Decimal("102.0")]
    assert isinstance(e.bids[0][1], Decimal) and e.bids[0][1] == Decimal("0.5")


def test_trade_sides():
    for s in ("buy", "sell"):
        e = to_engine_event(Trade(5, TOOBIT, "ETH-SWAP-USDT", 3000.5, 0.25, s, "1", 6))
        assert isinstance(e, TradeEvent) and e.side == s and e.instrument_id == ETH
        assert e.price == Decimal("3000.5") and e.size == Decimal("0.25")


def test_ctx_and_funding():
    c = to_engine_event(MarketCtx(7, TOOBIT, "BTC-SWAP-USDT", 8, funding_rate=0.0001, mark_price=100.5,
                                  index_price=100.4, next_funding_ms=9000, funding_interval_h=8.0))
    assert isinstance(c, CtxEvent) and c.mark == Decimal("100.5") and c.index == Decimal("100.4")
    assert c.funding_rate == Decimal("0.0001") and c.next_funding_ms == 9000
    f = to_engine_event(Funding(1, TOOBIT, "BTC-SWAP-USDT", 0.0002, 8.0, 2000, 3))
    assert isinstance(f, FundingEvent) and f.ts_ms == 2000 and f.rate == Decimal("0.0002")


def test_binance_and_others_none():
    assert to_engine_event(Trade(5, "binance_usdm", "BTCUSDT", 1.0, 1.0, "buy", "1", 6)) is None
    assert to_engine_event(BookSnapshot(1, "binance_usdm", "BTCUSDT", [[1, 1]], [[2, 1]], 2)) is None
    assert to_engine_event(Candle(0, TOOBIT, "BTC-SWAP-USDT", "1m", 1, 1, 1, 1, 1, 1)) is None

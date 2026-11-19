from decimal import Decimal

import pytest

from league.fill import BookEvent, InstrumentSpec
from league.fill.margin import hl_liquidation_price
from league.fill.preview import order_preview, touch_probability

BTC = InstrumentSpec("BTC-USDC-PERP", "hyperliquid", max_leverage=40)
UBTC = InstrumentSpec("UBTC-USDC-SPOT", "hyperliquid", market_type="SPOT")
BOOK = BookEvent(0, "BTC-USDC-PERP", [(99_990, 1), (99_980, 2)], [(100_010, 1), (100_020, 2)])


def test_market_buy_walks_book_and_costs():
    r = order_preview(BTC, "buy", "2", mark=100_000, equity=100_000, book=BOOK)
    assert r["order_placed"] is False
    assert Decimal(r["est_price"]) == Decimal("100015")          # 1 @ 100010 + 1 @ 100020
    assert Decimal(r["cost"]["fee"]) == Decimal("200030") * Decimal("0.00045")
    assert Decimal(r["cost"]["slippage_vs_mark"]) == Decimal("30")
    assert r["slippage_bps"] == 1.5                                 # 중간가 100000 대비


def test_liquidation_matches_hl_formula():
    r = order_preview(BTC, "buy", "2", mark=100_000, equity=10_000, book=BOOK)
    e1 = Decimal(10_000) - Decimal(r["cost"]["fee"]) - Decimal(30)
    l = Decimal(1) / 80
    ma = e1 - 2 * Decimal(100_000) * l
    want = hl_liquidation_price(Decimal(100_000), Decimal(2), 1, ma, l)
    assert abs(Decimal(r["liquidation_price"]) - want) < Decimal("1e-6")
    assert r["leverage_after"] == pytest.approx(float(200_000 / e1), rel=1e-3)   # 수수료·슬리피지 뒤 자산 기준


def test_short_liquidation_above_mark_and_touch_probability():
    r = order_preview(BTC, "sell", "1", mark=100_000, equity=20_000, book=BOOK, vol_annual=0.5, horizon_days=30)
    assert Decimal(r["liquidation_price"]) > 100_000
    assert 0 < r["liquidation_touch_probability"] < 1
    assert r["max_loss_basis"].startswith("liquidation")


def test_touch_probability_edges():
    assert touch_probability(Decimal(100), Decimal(100), 0.5, 1) == 1.0
    far = touch_probability(Decimal(100), Decimal(10), 0.5, 1)
    assert far < 1e-6


def test_stop_loss_and_sizing():
    r = order_preview(BTC, "buy", "1", mark=100_000, equity=100_000, book=BOOK, stop_price=95_000, loss_limit=1_000)
    loss = Decimal(r["max_loss"])
    assert Decimal(5010) < loss < Decimal(5010) + 100            # 가격 5,010 + 진입·청산 수수료
    sized = Decimal(r["size_for_loss_limit"])
    assert Decimal("0.19") < sized < Decimal("0.2")              # 1,000 / (5,010 + 수수료)


def test_limit_maker_vs_marketable():
    rest = order_preview(BTC, "buy", "1", mark=100_000, equity=100_000, book=BOOK, order_type="limit",
                         limit_price=99_000)
    assert rest["cost"]["liquidity"] == "maker" and Decimal(rest["cost"]["fee"]) == Decimal(99_000) * Decimal("0.00015")
    cross = order_preview(BTC, "buy", "1", mark=100_000, equity=100_000, book=BOOK, order_type="limit",
                          limit_price=100_010)
    assert "limit_marketable" in cross["flags"] and cross["cost"]["liquidity"] == "taker"


def test_depth_exhausted_and_over_leverage_flags():
    r = order_preview(BTC, "buy", "5", mark=100_000, equity=10_000, book=BOOK)
    assert "depth_exhausted" in r["flags"] and "over_max_leverage" in r["flags"]


def test_funding_per_day_sign():
    long = order_preview(BTC, "buy", "1", mark=100_000, equity=100_000, book=BOOK, funding_rate="0.0000125",
                         funding_interval_h=1)
    short = order_preview(BTC, "sell", "1", mark=100_000, equity=100_000, book=BOOK, funding_rate="0.0000125",
                          funding_interval_h=1)
    assert Decimal(long["cost"]["funding_per_day"]) == Decimal(30)      # 1 × 100,000 × 0.00125% × 24 → 롱 지급
    assert Decimal(short["cost"]["funding_per_day"]) == Decimal(-30)


def test_spot_has_no_liquidation():
    book = BookEvent(0, "UBTC-USDC-SPOT", [(99_990, 1)], [(100_010, 1)])
    r = order_preview(UBTC, "buy", "1", mark=100_000, equity=200_000, book=book)
    assert r["liquidation_price"] is None and r["cost"]["funding_per_day"] is None
    assert Decimal(r["cost"]["fee"]) == Decimal(100_010) * Decimal("0.0007")

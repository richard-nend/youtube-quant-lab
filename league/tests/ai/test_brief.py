from league.ai.brief import build_brief
from league.ai.prompt import render_prompt
from league.collect.models import BookSnapshot, MarketCtx
from league.collect.store import RawStore
from league.venues import TOOBIT

NOW = 1_800_000_000_000
LEVELS = 12


def fake_candles(symbol, interval, n):
    return [[NOW - (n - i) * 3_600_000, 100 + i, 110 + i, 90 + i, 105 + i, 1.0] for i in range(n)]


def make_store(tmp_path):
    s = RawStore(tmp_path)
    bids = [[100 - i, 1.0 + (i == 0) * 2] for i in range(LEVELS)]   # 상위 10단계 합 = 3+9 = 12, 11·12번째는 큰 값이라 무시돼야 함
    bids[10][1] = bids[11][1] = 50.0
    asks = [[101 + i, 1.0] for i in range(LEVELS)]                  # 상위 10단계 합 = 10
    for sym in ("BTC-SWAP-USDT", "ETH-SWAP-USDT"):
        s.append(BookSnapshot(NOW - 1000, TOOBIT, sym, bids, asks, NOW - 1000))
        s.append(MarketCtx(NOW - 2000, TOOBIT, sym, NOW - 2000, funding_rate=0.0001, mark_price=100.5, open_interest=777.0))
    s.append(BookSnapshot(NOW + 5000, TOOBIT, "BTC-SWAP-USDT", [[1, 1]], [[2, 1]], NOW + 5000))  # 미래 기록은 보지 않는다
    s.flush()
    return s


def test_counts_and_imbalance(tmp_path):
    b = build_brief(make_store(tmp_path), {"equity": 5000}, [], NOW, candles=fake_candles)
    btc = b["markets"]["BTC"]
    assert len(btc["candles_4h"]) == 60 and len(btc["candles_1h"]) == 48
    assert abs(btc["book_imbalance"] - 12 / 22) < 1e-9          # 12 / (12 + 10)
    assert btc["funding_rate"] == 0.0001 and btc["open_interest"] == 777.0
    assert b["account"] == {"equity": 5000} and b["history"] == []
    assert "rules" in b


def test_prompt_same_for_all_teams_and_korean(tmp_path):
    b = build_brief(make_store(tmp_path), {"equity": 5000}, [], NOW, candles=fake_candles)
    p = render_prompt(b)
    assert p == render_prompt(b)
    assert "JSON" in p and "reason" in p and "한국어" in p


def test_rules_text(tmp_path):
    r = build_brief(make_store(tmp_path), {}, [], NOW, candles=fake_candles)["rules"]
    assert "4시간마다" in r and "3배" in r and "합계" in r and "매시간" not in r

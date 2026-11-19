from league.collect.models import BookSnapshot, Trade, to_row
from league.collect.store import RawStore

H = 3_600_000
T0 = 1_761_400_000_000


def book(ts):
    return BookSnapshot(ts_ms=ts, venue="toobit_usdtm", symbol="BTC-SWAP-USDT", bids=[[100.0, 1.5], [99.0, 2.0]],
                        asks=[[101.0, 1.0]], recv_ms=ts + 5)


def trade(ts, i):
    return Trade(ts_ms=ts, venue="toobit_usdtm", symbol="BTC-SWAP-USDT", price=100.5, size=0.01, side="buy",
                 trade_id=str(i), recv_ms=ts + 3)


def test_store_roundtrip(tmp_path):
    s = RawStore(tmp_path)
    books = [book(T0 + 2 * H), book(T0), book(T0 + 1000)]  # 시간 걸침 + 순서 뒤섞음
    trades = [trade(T0 + 500, 1), trade(T0 + H, 2)]
    for r in books + trades:
        s.append(r)
    s.flush()
    got = s.read("toobit_usdtm", "book", T0, T0 + 3 * H)
    assert [r["ts_ms"] for r in got] == [T0, T0 + 1000, T0 + 2 * H]
    assert got[0] == to_row(books[1])
    assert [r["trade_id"] for r in s.read("toobit_usdtm", "trades", T0, T0 + 3 * H)] == ["1", "2"]
    assert [r["ts_ms"] for r in s.read("toobit_usdtm", "book", T0 + 1000, T0 + 2 * H)] == [T0 + 1000]
    assert s.read("binance_usdm", "book", T0, T0 + H) == []


def test_store_flush_twice_no_dupes(tmp_path):
    s = RawStore(tmp_path)
    s.append(book(T0)); s.append(book(T0 + 1))
    s.flush(); s.flush()
    assert len(s.read("toobit_usdtm", "book", T0, T0 + H)) == 2
    s.append(book(T0 + 2)); s.flush()  # 같은 시간에 다시 쓰면 새 seq 파일, 합쳐 3개
    assert len(s.read("toobit_usdtm", "book", T0, T0 + H)) == 3

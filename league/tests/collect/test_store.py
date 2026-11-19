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


def test_store_write_failure_keeps_rows(tmp_path, monkeypatch):
    import pyarrow.parquet as pq
    s = RawStore(tmp_path)
    s.append(book(T0)); s.append(book(T0 + 2 * H))  # 서로 다른 시간 = 버킷 둘
    real, calls = pq.write_table, []

    def flaky(*a, **k):
        calls.append(1)
        if len(calls) == 1:
            raise OSError("disk full")
        return real(*a, **k)

    monkeypatch.setattr(pq, "write_table", flaky)
    try:
        s.flush()
        assert False, "should raise"
    except OSError:
        pass
    monkeypatch.setattr(pq, "write_table", real)
    s.flush()
    assert [r["ts_ms"] for r in s.read("toobit_usdtm", "book", T0, T0 + 3 * H)] == [T0, T0 + 2 * H]


def test_store_seq_after_gap_not_overwritten(tmp_path):
    s = RawStore(tmp_path)
    s.append(book(T0)); s.flush()
    d = next(tmp_path.glob("venue=*/kind=book/date=*"))
    h = next(d.glob("hour=*-0.parquet")).name.split("-")[0]
    (d / f"{h}-2.parquet").write_bytes((d / f"{h}-0.parquet").read_bytes())  # 파일 -0, -2 (빈 번호 1)
    before = (d / f"{h}-2.parquet").read_bytes()
    s.append(book(T0 + 1)); s.flush()
    assert sorted(p.name for p in d.glob("*.parquet")) == [f"{h}-0.parquet", f"{h}-2.parquet", f"{h}-3.parquet"]
    assert (d / f"{h}-2.parquet").read_bytes() == before

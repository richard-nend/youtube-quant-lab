"""원시 기록 저장: 메모리 버퍼 → flush 때 시간별 parquet.

경로 root/venue=<v>/kind=<k>/date=YYYY-MM-DD/hour=HH-<seq>.parquet (UTC). flush 는 버퍼를 비우므로 두 번 불러도 중복 없음.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from .models import Record, to_row


class RawStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        self._buf: dict[tuple[str, str, str, str], list[dict]] = defaultdict(list)

    def append(self, rec: Record) -> None:
        dt = datetime.fromtimestamp(rec.ts_ms / 1000, tz=timezone.utc)
        self._buf[(rec.venue, rec.kind, dt.strftime("%Y-%m-%d"), dt.strftime("%H"))].append(to_row(rec))

    def flush(self) -> None:
        # 쓰기에 성공한 버킷만 버퍼에서 뺀다. 실패하면 나머지는 버퍼에 남기고 예외를 다시 던진다.
        for key in list(self._buf):
            venue, kind, date, hour = key
            rows = self._buf[key]
            d = self.root / f"venue={venue}" / f"kind={kind}" / f"date={date}"
            d.mkdir(parents=True, exist_ok=True)
            seqs = [int(p.stem.rsplit("-", 1)[1]) for p in d.glob(f"hour={hour}-*.parquet")]
            seq = max(seqs, default=-1) + 1
            tmp, final = d / f".hour={hour}-{seq}.tmp", d / f"hour={hour}-{seq}.parquet"
            pq.write_table(pa.Table.from_pylist(rows), tmp, compression="zstd")
            if final.exists():  # 같은 번호가 생겼으면 덮어쓰지 않는다
                tmp.unlink()
                raise FileExistsError(final)
            tmp.rename(final)  # 읽는 쪽이 쓰다 만 파일을 보지 않게
            del self._buf[key]

    def read(self, venue: str, kind: str, start_ms: int, end_ms: int) -> list[dict]:
        """start_ms <= ts_ms < end_ms, ts 오름차순(같으면 파일·행 순서 유지)."""
        out: list[dict] = []
        for f in sorted((self.root / f"venue={venue}" / f"kind={kind}").glob("date=*/hour=*.parquet")):
            out += [r for r in pq.read_table(f).to_pylist() if start_ms <= r["ts_ms"] < end_ms]
        return sorted(out, key=lambda r: r["ts_ms"])

"""리그 코어: 팀마다 체결 엔진 1개, 같은 이벤트를 모두에게 먹이고 스냅샷/복원."""
from __future__ import annotations

import logging
import os
import pickle
from decimal import Decimal
from pathlib import Path

from .fill import Engine
from .fill.engine import EngineView
from .venues import league_engine_config

TEAMS = ("claude", "gpt", "gemini", "ours")
SNAP_VERSION = 1   # 엔진·리그 구조가 바뀌어 옛 스냅샷을 못 쓰게 되면 올린다
log = logging.getLogger("league.core")


class QueueStrategy:
    """쌓인 주문을 다음 이벤트 처리 직후 엔진에 내보낸다."""

    def __init__(self):
        self.pending: list = []

    def push(self, items: list) -> None:
        self.pending.extend(items)

    def on_event(self, ev, view):
        out, self.pending = self.pending, []
        return out


class League:
    def __init__(self, teams: list[str] | tuple[str, ...] = TEAMS, start_ms: int = 0):
        self.teams = list(teams)
        self.last_ts = start_ms
        self.engines = {t: Engine(league_engine_config(Decimal(5000))) for t in self.teams}
        self.queues = {t: QueueStrategy() for t in self.teams}

    def step(self, ev) -> None:
        for t in self.teams:
            self.engines[t].step(ev, self.queues[t])
        self.last_ts = ev.at_ms

    def submit(self, team: str, items) -> None:
        self.queues[team].push(list(items))

    def view(self, team: str) -> EngineView:
        return EngineView(self.engines[team])

    def snapshot(self, path: Path) -> None:
        """임시 파일에 쓰고 fsync → 지금 것을 .prev 로 → 바꿔 넣기(반쯤 쓴 파일이 남지 않게, 직전 것 하나는 늘 남게)."""
        path = Path(path)
        self.snap_version = SNAP_VERSION
        data = pickle.dumps(self)
        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        if path.exists():
            os.replace(path, path.with_name(path.name + ".prev"))
        os.replace(tmp, path)
        fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    @classmethod
    def restore(cls, path: Path) -> "League":
        """스냅샷, 안 되면 .prev. 둘 다 안 되면 RuntimeError(이유 둘 다)."""
        path, errs = Path(path), []
        for p in (path, path.with_name(path.name + ".prev")):
            try:
                lg = pickle.loads(p.read_bytes())   # 우리가 쓴 파일만 읽는다
                if not isinstance(lg, cls) or getattr(lg, "snap_version", None) != SNAP_VERSION:
                    raise ValueError(f"not a league snapshot v{SNAP_VERSION}")
                if p != path:
                    log.error("snapshot %s unreadable (%s), restored %s", path.name, errs[0], p.name)
                return lg
            except Exception as e:
                errs.append(f"{p.name}: {e!r}"[:300])
        raise RuntimeError("; ".join(errs))

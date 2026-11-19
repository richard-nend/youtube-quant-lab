"""리그 코어: 팀마다 체결 엔진 1개, 같은 이벤트를 모두에게 먹이고 스냅샷/복원."""
from __future__ import annotations

import pickle
from decimal import Decimal
from pathlib import Path

from .fill import Engine
from .fill.engine import EngineView
from .venues import league_engine_config

TEAMS = ("claude", "gpt", "gemini", "ours")


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
        tmp = Path(str(path) + ".tmp")
        tmp.write_bytes(pickle.dumps(self))
        tmp.replace(path)  # 반쯤 쓴 파일이 남지 않게

    @classmethod
    def restore(cls, path: Path) -> "League":
        return pickle.loads(Path(path).read_bytes())

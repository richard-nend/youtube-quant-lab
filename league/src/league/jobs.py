"""예약 작업. `python -m league.jobs decide --data <dir>`: 세 AI 팀을 순서대로 판단(launchd, 4시간봉 마감 후 33분)."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .ai import decide as dec
from .collect.util import now_ms

log = logging.getLogger("league.jobs")
AI_TEAMS = ("claude", "gpt", "gemini")


def decide_all(root: Path, now: int | None = None, decide=None) -> None:
    """한 팀이 실패해도 다음 팀은 판단한다. 판단 파일을 데몬이 다음 분에 적용."""
    decide = decide or dec.decide
    now = now or now_ms()
    for team in AI_TEAMS:
        try:
            log.info("%s -> %s", team, decide(team, now, root=root))
        except Exception:
            log.exception("%s decide crashed", team)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="league.jobs")
    ap.add_argument("job", choices=["decide"])
    ap.add_argument("--data", required=True, type=Path)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    decide_all(a.data)


if __name__ == "__main__":
    main()

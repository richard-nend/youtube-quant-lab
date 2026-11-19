"""예약 작업. `python -m league.jobs decide --data <dir>`: 세 AI 팀을 같은 시장 요약으로 동시에 판단(launchd, 4시간봉 마감 후 33분)."""
from __future__ import annotations

import argparse
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .ai import decide as dec
from .ai.brief import build_markets
from .collect.store import RawStore
from .collect.util import now_ms

log = logging.getLogger("league.jobs")
AI_TEAMS = ("claude", "gpt", "gemini")


def decide_all(root: Path, now: int | None = None, decide=None, markets=None) -> None:
    """시장 요약을 한 번 만들고(같은 시장 자료) 세 팀을 동시에 묻는다(리뷰 I5). 한 팀이 실패해도 다른 팀은 판단한다.
    판단 파일을 데몬이 다음 분에 적용."""
    decide = decide or dec.decide
    now = now or now_ms()
    if markets is None:
        try:
            markets = build_markets(RawStore(Path(root) / "raw"), now)
        except Exception as e:
            log.exception("market summary failed: all teams hold this round")
            markets = e
    with ThreadPoolExecutor(len(AI_TEAMS)) as ex:
        futs = {t: ex.submit(decide, t, now, root=root, markets=markets) for t in AI_TEAMS}
    for team, f in futs.items():
        try:
            log.info("%s -> %s", team, f.result())
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

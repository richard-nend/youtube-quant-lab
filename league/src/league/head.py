"""돌고 있는 코드의 git 커밋(데몬 시작 로그·판단 파일에 남긴다: 버그 수정이 언제부터 적용됐는지 추적)."""
from __future__ import annotations

import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def git_head(repo: Path = REPO) -> str | None:
    try:
        r = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5)
        return r.stdout.strip() or None
    except Exception:
        return None

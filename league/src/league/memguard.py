"""무거운 모델 로드 직전 가용 메모리 확인(macOS vm_stat). 3GB 미만이면 그 회차를 건너뛴다."""
from __future__ import annotations

import re
import subprocess


def available_bytes(text: str | None = None) -> int:
    """(free + inactive + speculative) 페이지 × 페이지 크기. text 없으면 vm_stat 실행."""
    text = text if text is not None else subprocess.run(["vm_stat"], capture_output=True, text=True, check=True).stdout
    page = int(re.search(r"page size of (\d+) bytes", text).group(1))
    pages = lambda k: int(re.search(rf"Pages {k}:\s+(\d+)", text).group(1))
    return (pages("free") + pages("inactive") + pages("speculative")) * page


def enough_memory(min_bytes: int = 3 * 2**30) -> bool:
    return available_bytes() >= min_bytes

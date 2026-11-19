"""한 팀의 한 번 판단: 계좌 읽기 → 요약 → 호출(재시도 3, 형식 오류 1회 재질문) → 판단 파일. 실패는 hold_fallback."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from ..collect.store import RawStore
from ..decision import validate
from .brief import build_brief
from .callers import call_claude, call_codex, call_gemini
from .prompt import render_prompt, reask_prompt

CALLERS = {"claude": call_claude, "gpt": call_codex, "gemini": call_gemini}
DEFAULT_ROOT = Path(os.environ.get("LEAGUE_DATA", Path(__file__).resolve().parents[3] / "data"))


def parse_decision(text: str, team: str, utc: str) -> dict:
    """모델 답에서 JSON 객체를 꺼내 판단 파일 형태로. 형식이 틀리면 ValueError."""
    i, j = text.find("{"), text.rfind("}")
    if i < 0 or j < i:
        raise ValueError("no JSON object")
    d = json.loads(text[i:j + 1])
    if not isinstance(d, dict):
        raise ValueError("not an object")
    d = {"team": team, "decided_utc": utc, "status": "ok", "items": d.get("items")}
    errs = validate(d)
    if errs:
        raise ValueError("; ".join(errs[:5]))
    return d


def _history(dec_dir: Path, utc: str) -> list[dict]:
    out = []
    for f in sorted(dec_dir.glob("*.json"))[::-1]:
        if f.stem >= utc:
            continue
        d = json.loads(f.read_text())
        out.append({"decided_utc": d["decided_utc"], "status": d["status"], "result": d.get("result"),
                    "items": [{k: it.get(k) for k in ("symbol", "action", "notional_pct", "reason")} for it in d["items"]]})
        if len(out) == 3:
            break
    return out[::-1]


def decide(team: str, now_ms: int, *, root: Path | None = None, caller=None, store=None, candles=None) -> Path:
    root = Path(root or DEFAULT_ROOT)
    utc = datetime.fromtimestamp(now_ms / 1000, timezone.utc).strftime("%Y%m%dT%H%M")
    path = root / "decisions" / team / f"{utc}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    out, err = {"team": team, "decided_utc": utc, "status": "hold_fallback", "items": []}, ""
    try:
        acct = json.loads((root / "state" / f"account_{team}.json").read_text())
    except (OSError, ValueError) as e:
        acct, err = None, f"account state unreadable: {e}"
    if acct is not None:
        kw = {"candles": candles} if candles else {}
        prompt = render_prompt(build_brief(store or RawStore(root / "raw"), acct, _history(path.parent, utc), now_ms, **kw))
        caller, p, reasked = caller or CALLERS[team], prompt, False
        for _ in range(3):
            try:
                res = caller(p)
                d = parse_decision(res.text, team, utc)
            except ValueError as e:   # 형식 오류: 한 번만 재질문
                err = f"format: {e}"
                if not reasked:
                    reasked, p = True, reask_prompt(prompt, str(e)[:200])
                continue
            except Exception as e:
                err = f"call: {type(e).__name__}: {str(e)[:200]}"
                continue
            out, err = {**d, "model": res.model, "model_version": res.model_version,
                        "tokens_in": res.tokens_in, "tokens_out": res.tokens_out}, ""
            break
    if err:
        out["error"] = err
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    tmp.replace(path)
    return path

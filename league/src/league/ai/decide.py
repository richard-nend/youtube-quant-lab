"""한 팀의 한 번 판단: 계좌 읽기 → 요약 → 호출(실패 시 30초·120초 뒤 다시, 최대 3번; 형식 오류는 따로 1회 재질문)
→ 판단 파일. 실패는 hold_fallback."""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from ..collect.store import RawStore
from ..decision import validate
from ..head import git_head
from .brief import build_brief
from .callers import call_claude, call_codex, call_gemini
from .prompt import render_prompt, reask_prompt

CALLERS = {"claude": call_claude, "gpt": call_codex, "gemini": call_gemini}
BACKOFF_S = (30, 120)
ACCOUNT_MAX_AGE_MS = 10 * 60_000   # 계좌 파일은 매분 갱신   # 호출 실패 뒤 대기(초): 3번 시도
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


def _account(root: Path, team: str, now_ms: int) -> tuple[dict | None, str]:
    try:
        acct = json.loads((root / "state" / f"account_{team}.json").read_text())
        asof = datetime.strptime(acct["as_of_utc"], "%Y-%m-%dT%H:%MZ").replace(tzinfo=timezone.utc).timestamp() * 1000
    except (OSError, ValueError, KeyError, TypeError) as e:
        return None, f"account state unreadable: {e}"
    if now_ms - asof > ACCOUNT_MAX_AGE_MS:   # 데몬이 멈췄다: 옛 계좌로 판단하지 않는다
        return None, f"account state stale (as of {acct['as_of_utc']})"
    return acct, ""


def decide(team: str, now_ms: int, *, root: Path | None = None, caller=None, store=None, candles=None,
           sleep=time.sleep, markets=None) -> Path:
    """markets = 회차에서 한 번 만든 시장 요약(세 팀 공통). 예외 객체면 그 회차 요약 실패 → 유지."""
    root = Path(root or DEFAULT_ROOT)
    utc = datetime.fromtimestamp(now_ms / 1000, timezone.utc).strftime("%Y%m%dT%H%M")
    path = root / "decisions" / team / f"{utc}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    out, err = {"team": team, "decided_utc": utc, "status": "hold_fallback", "items": []}, ""
    acct, err = _account(root, team, now_ms)
    use = {"model": None, "model_version": None, "tokens_in": 0, "tokens_out": 0}   # 재시도·재질문 포함 전체 합계
    if acct is not None:
        try:
            if isinstance(markets, Exception):
                raise markets
            kw = {k: v for k, v in (("candles", candles), ("markets", markets)) if v is not None}
            prompt = render_prompt(build_brief(store or RawStore(root / "raw"), acct, _history(path.parent, utc), now_ms, **kw))
        except Exception as e:
            prompt, err = None, f"brief: {type(e).__name__}: {str(e)[:200]}"
        caller, p, reasked, fails = caller or CALLERS[team], prompt, False, 0
        while prompt and fails < len(BACKOFF_S) + 1:
            try:   # 호출 쪽 예외는 종류와 상관없이 호출 실패(CLI 출력 해석의 ValueError 포함)
                res = caller(p)
            except Exception as e:
                err, fails = f"call: {type(e).__name__}: {str(e)[:200]}", fails + 1
                if fails <= len(BACKOFF_S):
                    sleep(BACKOFF_S[fails - 1])   # 간격을 두고 다시(설계 4절)
                continue
            use["model"], use["model_version"] = res.model, res.model_version
            use["tokens_in"] += res.tokens_in
            use["tokens_out"] += res.tokens_out
            try:
                out, err = parse_decision(res.text, team, utc), ""
                break
            except ValueError as e:   # 형식 오류: 호출 재시도와 따로, 한 번만 재질문
                err = f"format: {e}"
                if reasked:
                    break
                reasked, p = True, reask_prompt(prompt, str(e)[:200])
    out.update({k: v for k, v in use.items() if v or k.startswith("tokens")})
    out["code_head"] = git_head()
    if err:
        out["error"] = err
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    tmp.replace(path)
    return path

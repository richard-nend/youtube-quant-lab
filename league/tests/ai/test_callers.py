import json
from pathlib import Path

import pytest

from league.ai import callers, decide as dmod
from league.ai.callers import CallResult
from league.decision import validate

NOW = 1_800_000_000_000
GOOD = json.dumps({"items": [{"symbol": "BTC", "action": "long", "notional_pct": 100, "order": "market", "reason": "상승"}]})


def res(text):
    return CallResult(text, "m", "m-1", 10, 5)


@pytest.fixture
def env(tmp_path, monkeypatch):
    (tmp_path / "state").mkdir()
    (tmp_path / "state" / "account_claude.json").write_text(json.dumps({"equity": 5000, "as_of_utc": "2027-01-15T08:00Z"}))
    monkeypatch.setattr(dmod, "build_brief", lambda store, account, history, now_ms, **k: {"account": account, "history": history})
    monkeypatch.setattr(dmod, "render_prompt", lambda b: "PROMPT")
    return tmp_path


SLEPT = []


def run(env, fake):
    SLEPT.clear()
    p = dmod.decide("claude", NOW, root=env, caller=fake, sleep=SLEPT.append)
    return p, json.loads(p.read_text())


def test_ok(env):
    p, d = run(env, lambda prompt: res(f"```json\n{GOOD}\n```"))
    assert d["status"] == "ok" and not validate(d) and d["tokens_in"] == 10 and d["model_version"] == "m-1"
    assert p.parent == env / "decisions" / "claude"


def test_retry_on_exception_then_ok(env):
    calls = []

    def fake(prompt):
        calls.append(prompt)
        if len(calls) < 3:
            raise RuntimeError("boom")
        return res(GOOD)
    _, d = run(env, fake)
    assert d["status"] == "ok" and len(calls) == 3 and SLEPT == [30, 120]   # 호출 실패 사이에 간격(리뷰 I6)


def test_three_call_failures_hold_with_backoff(env):
    calls = []

    def fake(prompt):
        calls.append(1)
        raise ValueError("cli printed garbage")   # 호출 쪽 ValueError 는 형식 오류가 아니라 호출 실패
    _, d = run(env, fake)
    assert d["status"] == "hold_fallback" and len(calls) == 3 and SLEPT == [30, 120] and d["error"].startswith("call:")


def test_reask_does_not_use_up_call_retries(env):
    seq = iter(["bad", RuntimeError("x"), RuntimeError("y"), GOOD])

    def fake(prompt):
        r = next(seq)
        if isinstance(r, Exception):
            raise r
        return res(r)
    _, d = run(env, fake)
    assert d["status"] == "ok"


def test_vertex_non_json_body_is_call_failure(monkeypatch):
    class R:
        status_code = 502
        text = "<html>bad gateway</html>"

        def json(self):
            raise ValueError("Expecting value")

    class Auth:
        base = "https://x"

        def __init__(self, *a):
            pass

        def headers(self):
            return {}
    monkeypatch.setattr(callers, "_VertexAuth", Auth)
    monkeypatch.setattr(callers.requests, "post", lambda *a, **k: R())
    with pytest.raises(RuntimeError, match="502"):
        callers._vertex("p", "m")


def test_vertex_output_tokens_include_thinking(monkeypatch):
    monkeypatch.setattr(callers, "_vertex", lambda prompt, model: (200, {
        "candidates": [{"content": {"parts": [{"text": "t", "thought": True}, {"text": "ans"}]}}],
        "usageMetadata": {"promptTokenCount": 7, "candidatesTokenCount": 3, "thoughtsTokenCount": 40}}))
    r = callers.call_gemini("p")
    assert (r.text, r.tokens_in, r.tokens_out) == ("ans", 7, 43)


def test_bad_format_reasks_once(env):
    calls = []

    def fake(prompt):
        calls.append(prompt)
        return res("not json" if len(calls) == 1 else GOOD)
    _, d = run(env, fake)
    assert d["status"] == "ok" and len(calls) == 2 and calls[1].startswith("PROMPT") and calls[1] != "PROMPT"


def test_all_fail_hold_fallback(env):
    calls = []
    _, d = run(env, lambda prompt: calls.append(1) or res("nope"))
    assert d["status"] == "hold_fallback" and d["items"] == [] and d["error"] and not validate(d)
    assert len(calls) == 2 and SLEPT == []   # 형식 오류: 재질문 1회뿐


def test_brief_failure_hold_fallback(env, monkeypatch):
    def boom(*a, **k):
        raise OSError("candle fetch failed")
    monkeypatch.setattr(dmod, "build_brief", boom)
    _, d = run(env, lambda prompt: pytest.fail("must not call"))
    assert d["status"] == "hold_fallback" and "candle fetch failed" in d["error"] and not validate(d)


def test_corrupt_history_hold_fallback(env):
    h = env / "decisions" / "claude"
    h.mkdir(parents=True)
    (h / "20200101T0000.json").write_text("{corrupt")
    _, d = run(env, lambda prompt: pytest.fail("must not call"))
    assert d["status"] == "hold_fallback" and d["error"].startswith("brief:")


def test_tokens_summed_across_attempts_and_on_fallback(env):
    _, d = run(env, lambda prompt: res("nope"))     # 처음 + 재질문 = 2번 호출, 호출당 10/5
    assert d["status"] == "hold_fallback" and (d["tokens_in"], d["tokens_out"]) == (20, 10) and d["model"] == "m"
    calls = []

    def fake(prompt):
        calls.append(1)
        return res("bad" if len(calls) == 1 else GOOD)
    _, d = run(env, fake)
    assert d["status"] == "ok" and (d["tokens_in"], d["tokens_out"]) == (20, 10)


def test_403_does_not_fall_back(monkeypatch):
    monkeypatch.setattr(callers, "_vertex", lambda prompt, model: (403, {"error": {"message": "denied"}}))
    monkeypatch.setattr(callers, "_run", lambda *a, **k: pytest.fail("must not call agy"))
    with pytest.raises(RuntimeError, match="403"):
        callers.call_gemini("p")


def test_missing_account_hold_fallback(env):
    (env / "state" / "account_claude.json").unlink()
    _, d = run(env, lambda prompt: pytest.fail("must not call"))
    assert d["status"] == "hold_fallback" and "account" in d["error"]


def test_parse_claude_json():
    out = json.dumps({"result": "hi", "usage": {"input_tokens": 2, "cache_creation_input_tokens": 3, "cache_read_input_tokens": 4, "output_tokens": 7},
                      "modelUsage": {"claude-opus-5-5": {}}})
    r = callers.parse_claude(out, "claude-opus-5-5")
    assert (r.text, r.model_version, r.tokens_in, r.tokens_out) == ("hi", "claude-opus-5-5", 9, 7)


def test_parse_codex_events():
    out = "\n".join(json.dumps(e) for e in [
        {"type": "thread.started"}, {"type": "item.completed", "item": {"type": "agent_message", "text": "hello"}},
        {"type": "turn.completed", "usage": {"input_tokens": 11, "cached_input_tokens": 1, "output_tokens": 3}}])
    r = callers.parse_codex(out, "gpt-x")
    assert (r.text, r.tokens_in, r.tokens_out) == ("hello", 11, 3)


def test_gemini_falls_back_to_agy_on_quota(monkeypatch):
    monkeypatch.setattr(callers, "_vertex", lambda prompt, model: (429, {"error": {"status": "RESOURCE_EXHAUSTED"}}))
    monkeypatch.setattr(callers, "_run", lambda cmd, stdin=None, cwd=None: json.dumps(
        {"status": "SUCCESS", "response": "ok", "usage": {"input_tokens": 5, "output_tokens": 2}}))
    r = callers.call_gemini("p")
    assert r.text == "ok" and r.model.startswith("agy:")


TEAMS = ["call_claude", "call_codex", "call_gemini", "call_agy"]


@pytest.mark.live
@pytest.mark.parametrize("name", TEAMS)
def test_cli_tools_disabled(name, tmp_path):
    """CLI가 실제로 실행되는 폴더(cwd)에 카나리 파일을 둔다. 보이면 도구가 막히지 않은 것."""
    (tmp_path / "CANARY_LIVE_FILE_7q.txt").write_text("x")
    r = getattr(callers, name)("현재 폴더의 파일 목록을 알려줘", cwd=str(tmp_path))
    if "CANARY_LIVE_FILE_7q" in r.text:
        if name == "call_agy":
            pytest.xfail("agy cannot disable tools; it lists files in its cwd")
        pytest.fail(f"{name}: tools not blocked")


@pytest.mark.live
@pytest.mark.parametrize("name", TEAMS)
def test_live_team(tmp_path, name):
    from league.ai.brief import build_brief
    from league.ai.prompt import render_prompt
    from league.collect.store import RawStore
    now = 1_800_000_000_000
    prompt = render_prompt(build_brief(RawStore(tmp_path), {"equity": 5000, "positions": [], "open_orders": []}, [], now,
                                        candles=lambda s, i, n: [[now - (n - k) * 3_600_000, 100, 110, 90, 105, 1.0] for k in range(n)]))
    r = getattr(callers, name)(prompt)
    d = dmod.parse_decision(r.text, name, "20261119T0000")
    assert d and not validate(d), (name, r.text[:300])
    print("TOKENS", name, r.model, r.model_version, r.tokens_in, r.tokens_out)


def test_equal_reasoning_level_pinned_for_all_teams(monkeypatch):
    """사용자 결정(11-20): 세 팀 같은 생각 수준(high)을 명시. codex 사용자 설정(low)을 물려받지 않는다."""
    cmds = []

    def fake_run(cmd, stdin=None, cwd=None):
        cmds.append(cmd)
        if cmd[0] == "claude":
            return json.dumps({"result": "x", "usage": {"input_tokens": 1, "output_tokens": 1}})
        if cmd[0] == "agy":
            return json.dumps({"status": "SUCCESS", "response": "x"})
        return ""
    monkeypatch.setattr(callers, "_run", fake_run)
    callers.call_claude("p")
    callers.call_codex("p")
    callers.call_agy("p")
    claude, codex, agy = cmds
    assert claude[claude.index("--effort") + 1] == "high"
    assert codex[codex.index("-c") + 1] == 'model_reasoning_effort="high"'
    assert agy[agy.index("--model") + 1] == "gemini-4-argon-high"
    assert callers.vertex_body("p")["generationConfig"]["thinkingConfig"]["thinkingLevel"] == "HIGH"
    assert callers.CFG["gemini_vertex"] == "gemini-4-argon"


def test_stale_account_holds(env):
    """데몬이 멈춰 계좌 파일이 오래되면 옛 계좌로 판단하지 않고 유지."""
    (env / "state" / "account_claude.json").write_text(json.dumps({"equity": 5000, "as_of_utc": "2027-01-15T07:40Z"}))
    _, d = run(env, lambda prompt: pytest.fail("must not call"))
    assert d["status"] == "hold_fallback" and "stale" in d["error"]

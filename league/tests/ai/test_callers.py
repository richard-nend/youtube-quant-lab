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
    (tmp_path / "state" / "account_claude.json").write_text(json.dumps({"equity": 5000}))
    monkeypatch.setattr(dmod, "build_brief", lambda store, account, history, now_ms, **k: {"account": account, "history": history})
    monkeypatch.setattr(dmod, "render_prompt", lambda b: "PROMPT")
    return tmp_path


def run(env, fake):
    p = dmod.decide("claude", NOW, root=env, caller=fake)
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
    assert d["status"] == "ok" and len(calls) == 3


def test_bad_format_reasks_once(env):
    calls = []

    def fake(prompt):
        calls.append(prompt)
        return res("not json" if len(calls) == 1 else GOOD)
    _, d = run(env, fake)
    assert d["status"] == "ok" and len(calls) == 2 and calls[1].startswith("PROMPT") and calls[1] != "PROMPT"


def test_all_fail_hold_fallback(env):
    _, d = run(env, lambda prompt: res("nope"))
    assert d["status"] == "hold_fallback" and d["items"] == [] and d["error"] and not validate(d)


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
    _, d = run(env, lambda prompt: res("nope"))     # 3번 호출, 호출당 10/5
    assert d["status"] == "hold_fallback" and (d["tokens_in"], d["tokens_out"]) == (30, 15) and d["model"] == "m"
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

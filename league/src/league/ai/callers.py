"""AI 호출 경로 셋: claude / codex / gemini(Vertex, 한도 오류면 agy). 모델은 config/models.json.

CLI 판본·옵션 (2026-11-19 `--help` 확인):
- claude 2.1.328: `-p` 비대화형, `--tools ""` 내장 도구 전부 끔, `--no-session-persistence`, `--disable-slash-commands`,
  `--strict-mcp-config`(MCP 없음), `--setting-sources ""`(사용자 훅·설정 안 읽음), `--output-format json`.
- codex-cli 0.169.0: `codex exec`, `-s read-only`, `--ephemeral`, `--ignore-user-config`는 인증 영향 우려로 안 씀,
  `--disable shell_tool --disable unified_exec`로 셸 도구 끔(도구 차단 전용 옵션은 없음), `--json` 이벤트 출력, `-C <빈 임시폴더>`.
- agy 1.5.12: `-p=<프롬프트>`(값을 붙여야 함), `--output-format json`, `--sandbox`. 도구를 끄는 옵션이 없다
  (`--settings` permissions.deny, 빈 에이전트 정의 모두 효과 없음을 실험으로 확인) → 빈 임시폴더에서만 실행. 읽기 도구는 살아 있다.
- gemini 0.66.0 CLI는 쓰지 않는다(Vertex REST는 quant_lab.llm 의 인증 재사용).
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import requests
from quant_lab.llm import _VertexAuth

CFG = json.loads((Path(__file__).resolve().parents[3] / "config" / "models.json").read_text())
BIN_PATH = os.pathsep.join([str(Path.home() / ".local/bin"), "/opt/homebrew/bin", os.environ.get("PATH", "")])


@dataclass
class CallResult:
    text: str
    model: str
    model_version: str
    tokens_in: int
    tokens_out: int


def _run(cmd: list[str], stdin: str | None = None, cwd: str | None = None) -> str:
    """기본은 빈 임시폴더에서 실행(읽을 파일이 없게). cwd는 시험용. 실패는 예외."""
    with tempfile.TemporaryDirectory() as tmp:
        r = subprocess.run(cmd, input=stdin, capture_output=True, text=True, cwd=cwd or tmp, timeout=CFG["timeout_s"],
                           env={**os.environ, "PATH": BIN_PATH})
    if r.returncode:
        raise RuntimeError(f"{cmd[0]} exit {r.returncode}: {(r.stdout + r.stderr)[-300:]}")
    return r.stdout


def parse_claude(out: str, model: str) -> CallResult:
    d = json.loads(out)
    if d.get("is_error"):
        raise RuntimeError(f"claude error: {str(d.get('result'))[:200]}")
    u = d["usage"]
    tin = u["input_tokens"] + u.get("cache_creation_input_tokens", 0) + u.get("cache_read_input_tokens", 0)
    return CallResult(d["result"], model, next(iter(d.get("modelUsage") or {model: 0})), tin, u["output_tokens"])


def parse_codex(out: str, model: str) -> CallResult:
    text, usage = "", {}
    for line in out.splitlines():
        try:
            e = json.loads(line)
        except ValueError:
            continue
        if e.get("type") == "item.completed" and e["item"].get("type") == "agent_message":
            text = e["item"]["text"]   # 마지막 메시지가 답
        elif e.get("type") == "turn.completed":
            usage = e.get("usage", {})
        elif e.get("type") in ("error", "turn.failed"):
            raise RuntimeError(f"codex error: {json.dumps(e, ensure_ascii=False)[:300]}")
    return CallResult(text, model, model, usage.get("input_tokens", 0), usage.get("output_tokens", 0))


def call_claude(prompt: str, cwd: str | None = None) -> CallResult:
    m = CFG["claude"]
    out = _run(["claude", "-p", "--model", m, "--tools", "", "--no-session-persistence", "--disable-slash-commands",
                "--strict-mcp-config", "--setting-sources", "", "--output-format", "json"], prompt, cwd)
    return parse_claude(out, m)


def call_codex(prompt: str, cwd: str | None = None) -> CallResult:
    m = CFG["codex"]
    out = _run(["codex", "exec", "--ephemeral", "--skip-git-repo-check", "-s", "read-only", "-m", m,
                "--disable", "shell_tool", "--disable", "unified_exec", "--color", "never", "--json", "-"], prompt, cwd)
    return parse_codex(out, m)


def _vertex(prompt: str, model: str) -> tuple[int, dict]:
    v = _VertexAuth(os.environ.get("VERTEX_PROJECT") or CFG["vertex_project"], "global")
    r = requests.post(f"{v.base}/{model}:generateContent", headers=v.headers(), timeout=CFG["timeout_s"],
                      json={"contents": [{"role": "user", "parts": [{"text": prompt}]}]})
    return r.status_code, r.json()


def call_agy(prompt: str, cwd: str | None = None) -> CallResult:
    m = CFG["gemini_agy"]
    d = json.loads(_run(["agy", "--model", m, "--output-format", "json", "--sandbox", "--disable-slash-commands", f"-p={prompt}"], cwd=cwd))
    if d.get("status") != "SUCCESS":
        raise RuntimeError(f"agy status {d.get('status')}")
    u = d.get("usage", {})
    return CallResult(d["response"], f"agy:{m}", m, u.get("input_tokens", 0), u.get("output_tokens", 0))


def call_gemini(prompt: str) -> CallResult:
    m = CFG["gemini_vertex"]
    status, d = _vertex(prompt, m)
    if status in (402, 429) or "RESOURCE_EXHAUSTED" in json.dumps(d.get("error", {})):   # 크레딧·한도 → agy
        return call_agy(prompt)
    if status >= 400:
        raise RuntimeError(f"vertex {status}: {json.dumps(d, ensure_ascii=False)[:300]}")
    parts = d["candidates"][0]["content"]["parts"]
    u = d.get("usageMetadata", {})
    return CallResult("".join(p.get("text", "") for p in parts if not p.get("thought")), f"vertex:{m}",
                      d.get("modelVersion", m), u.get("promptTokenCount", 0), u.get("candidatesTokenCount", 0))

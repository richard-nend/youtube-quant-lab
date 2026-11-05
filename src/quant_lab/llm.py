"""Minimal Gemini REST client for experiments (EP02, EP05).

- Key comes from GEMINI_API_KEY (environment, else the project's .env). Never logged.
- Structured JSON output via responseSchema.
- Measures wall-clock latency of each call (this is the number quoted on screen).
- Retries on 429/5xx with backoff; every call is appended to a JSONL log.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import requests

from quant_lab import PROJECT_ROOT

API = "https://generativelanguage.googleapis.com/v1beta"


def _api_key() -> str:
    """Project .env wins over the shell environment (a stale global key once shadowed it)."""
    key = None
    env = PROJECT_ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if line.strip().startswith("GEMINI_API_KEY="):
                key = line.split("=", 1)[1].strip().strip('"').strip("'")
                break
    key = key or os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY not set")
    return key


def list_models() -> list[dict]:
    r = requests.get(f"{API}/models", params={"key": _api_key(), "pageSize": 200}, timeout=30)
    r.raise_for_status()
    return r.json().get("models", [])


class _VertexAuth:
    """Application Default Credentials (gcloud auth application-default login)."""

    def __init__(self, project: str | None, location: str):
        import google.auth
        from google.auth.transport.requests import Request

        self._creds, adc_project = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        self._request = Request()
        self.project = project or os.environ.get("VERTEX_PROJECT") or adc_project
        if not self.project:
            raise RuntimeError("set VERTEX_PROJECT (GCP project id linked to the billing account)")
        self.location = location
        host = "aiplatform.googleapis.com" if location == "global" else f"{location}-aiplatform.googleapis.com"
        self.base = f"https://{host}/v1/projects/{self.project}/locations/{location}/publishers/google/models"

    def headers(self) -> dict:
        if not self._creds.valid:
            self._creds.refresh(self._request)
        return {"Authorization": f"Bearer {self._creds.token}", "x-goog-user-project": self.project}


class Gemini:
    """backend: "vertex" (bills GCP credits) or "aistudio" (bills AI Studio prepay).
    Default comes from QL_LLM_BACKEND, else "vertex"."""

    def __init__(self, model: str, log_path: Path | None = None, temperature: float = 1.0,
                 thinking_budget: int | None = None, backend: str | None = None,
                 project: str | None = None, location: str = "global"):
        self.model = model
        self.log_path = log_path
        self.temperature = temperature
        self.thinking_budget = thinking_budget
        self.backend = backend or os.environ.get("QL_LLM_BACKEND", "vertex")
        if self.backend == "vertex":
            self._vertex = _VertexAuth(project, location)
            self._url = f"{self._vertex.base}/{model}:generateContent"
        else:
            self._key = _api_key()
            self._url = f"{API}/models/{model}:generateContent"

    def _post(self, body: dict):
        if self.backend == "vertex":
            return requests.post(self._url, headers=self._vertex.headers(), json=body, timeout=180)
        return requests.post(self._url, params={"key": self._key}, json=body, timeout=180)

    def json(self, system: str, user: str, schema: dict, *, tag: str = "", max_retries: int = 12) -> dict:
        body = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {
                "temperature": self.temperature,
                "responseMimeType": "application/json",
                "responseSchema": schema,
            },
        }
        if self.thinking_budget is not None:
            body["generationConfig"]["thinkingConfig"] = {"thinkingBudget": self.thinking_budget}
        delay = 5.0
        last = "no attempt"
        for attempt in range(max_retries):
            t0 = time.perf_counter()
            try:
                r = self._post(body)
            except (requests.Timeout, requests.ConnectionError) as e:
                last = type(e).__name__
                time.sleep(delay)
                delay = min(delay * 2, 120)
                continue
            latency = time.perf_counter() - t0
            if r.status_code in (429, 500, 502, 503, 504):
                last = f"{r.status_code}: {r.text[:160]}"
                time.sleep(delay)
                delay = min(delay * 2, 120)
                continue
            if r.status_code >= 400:
                raise RuntimeError(f"{self.backend} {r.status_code}: {r.text[:400]}")
            data = r.json()
            text = data["candidates"][0]["content"]["parts"][-1]["text"]
            out = json.loads(text)
            usage = data.get("usageMetadata", {})
            if self.log_path:
                self.log_path.parent.mkdir(parents=True, exist_ok=True)
                with self.log_path.open("a") as f:
                    f.write(json.dumps({"tag": tag, "model": self.model, "backend": self.backend,
                                        "latency_s": round(latency, 3),
                                        "failed_attempts": attempt,
                                        "usage": usage, "system": system, "user": user,
                                        "response": out}, ensure_ascii=False) + "\n")
            return {"response": out, "latency_s": latency, "usage": usage}
        raise RuntimeError(f"gave up after {max_retries} retries (last: {last})")

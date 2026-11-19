import json
import threading
from datetime import datetime, timezone

from league import jobs
from league.ai import brief as br, decide as dmod
from league.ai.callers import CallResult
from league.ai.prompt import HEAD
from league.jobs import decide_all

NOW = 1_800_000_000_000


def test_decide_all_one_crash_does_not_stop(tmp_path):
    seen = []

    def fake(team, now, root, markets=None):
        seen.append((team, now, root))
        if team == "claude":
            raise RuntimeError("boom")

    decide_all(tmp_path, 123, decide=fake, markets={})
    assert sorted(seen) == sorted([("claude", 123, tmp_path), ("gpt", 123, tmp_path), ("gemini", 123, tmp_path)])


def test_teams_get_same_market_snapshot_concurrently(tmp_path, monkeypatch):
    """리뷰 I5: 시장 요약은 회차당 한 번 만들어 세 팀에 같은 것을, 세 팀은 동시에 묻는다(각자 자기 계좌)."""
    fetched = []

    def candles(symbol, interval, n):
        fetched.append((symbol, interval, len(fetched)))   # 부를 때마다 다른 값 → 팀마다 따로 부르면 달라진다
        return [[NOW - (n - i) * 3_600_000, 100 + len(fetched), 110, 90, 105, 1.0] for i in range(n)]
    monkeypatch.setattr(br, "fetch_candles", candles)
    (tmp_path / "state").mkdir()
    asof = datetime.fromtimestamp(NOW / 1000, timezone.utc).strftime("%Y-%m-%dT%H:%MZ")
    for t in jobs.AI_TEAMS:
        (tmp_path / "state" / f"account_{t}.json").write_text(json.dumps({"as_of_utc": asof, "equity": t}))
    prompts, gate = {}, threading.Barrier(3, timeout=10)   # 셋이 동시에 있어야 통과

    def caller(team):
        def call(prompt):
            gate.wait()
            prompts[team] = json.loads(prompt[len(HEAD):])
            return CallResult('{"items":[]}', "m", "m", 1, 1)
        return call
    monkeypatch.setattr(dmod, "CALLERS", {t: caller(t) for t in jobs.AI_TEAMS})
    decide_all(tmp_path, NOW)
    assert len(fetched) == 4                                       # BTC·ETH × 4h·1h 한 번씩
    mk = [prompts[t]["markets"] for t in jobs.AI_TEAMS]
    assert mk[0] == mk[1] == mk[2]
    assert [prompts[t]["account"]["equity"] for t in jobs.AI_TEAMS] == list(jobs.AI_TEAMS)
    for t in jobs.AI_TEAMS:
        d = json.loads(next((tmp_path / "decisions" / t).glob("*.json")).read_text())
        assert d["status"] == "ok" and "code_head" in d

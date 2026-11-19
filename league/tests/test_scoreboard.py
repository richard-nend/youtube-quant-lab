import csv
import json
from decimal import Decimal

from league.core import League
from league.fill import BookEvent, CtxEvent, OrderRequest
from league.scoreboard import baseline_equity, build_scoreboard, write_public
from league.venues import BTC

MIDS = {"start": Decimal(100), "now": Decimal(110)}


def book(lg, t, bid, ask):
    lg.step(BookEvent(t, BTC, [(Decimal(bid), Decimal(500))], [(Decimal(ask), Decimal(500))]))


def decision(root, team, utc, status="ok", reason="r"):
    p = root / team
    p.mkdir(parents=True, exist_ok=True)
    items = [] if status == "hold_fallback" else [{"symbol": "BTC", "action": "hold", "reason": reason}]
    d = {"team": team, "decided_utc": utc, "status": status, "items": items}
    if status != "ok":
        d["error"] = "call: boom"
    (p / f"{utc}.json").write_text(json.dumps(d))


def make(tmp_path):
    lg = League(["claude", "gpt", "gemini", "ours"], 0)
    book(lg, 0, 99, 100)
    lg.submit("claude", [OrderRequest(BTC, "buy", "market", Decimal(1))])    # 수수료만큼 손해
    book(lg, 1000, 99, 100)
    book(lg, 2000, 99, 100)
    lg.submit("gpt", [OrderRequest(BTC, "buy", "market", Decimal(100))])     # 청산될 크기(명목 10000 > 5000, 하락)
    book(lg, 3000, 99, 100)
    book(lg, 4000, 99, 100)
    book(lg, 5000, 30, 31)
    lg.step(CtxEvent(5500, BTC, mark=Decimal(30)))                          # 마크 급락 → 청산 판정
    book(lg, 6000, 30, 31)
    book(lg, 7000, 30, 31)
    d = tmp_path / "dec"
    decision(d, "claude", "20261119T0000", reason="old")
    decision(d, "claude", "20261119T0100", status="hold_fallback")
    decision(d, "gemini", "20261119T0000", reason="g")
    return lg, d


def test_baseline_formula():
    assert baseline_equity(Decimal(110), Decimal(100), Decimal("0.0006")) == Decimal(5000) * Decimal("1.1") * (1 - Decimal("0.0006"))


def test_board(tmp_path):
    lg, d = make(tmp_path)
    b = build_scoreboard(lg, d, MIDS)
    assert b["simulated"] is True and b["engine_source"] and b["as_of_utc"]
    t = b["teams"]
    eq = [Decimal(r["equity"]) for r in t if not r["liquidated"]]
    assert eq == sorted(eq, reverse=True)
    assert t[-1]["team"] == "gpt" and t[-1]["liquidated"] is True      # 청산 팀 맨 아래
    assert all(not r["liquidated"] for r in t[:-1])
    c = next(r for r in t if r["team"] == "claude")
    assert c["fail_count"] == 1 and c["last_reason"]                    # 폴백 판단 = 마지막
    assert abs(sum(c["grade_ratio"].values()) - 1) < 1e-9
    assert next(r for r in t if r["team"] == "gemini")["last_reason"] == "g"
    assert next(r for r in t if r["team"] == "ours")["grade_ratio"] == {}   # 체결 없음
    assert b["baseline"]["equity"] == str(baseline_equity(Decimal(110), Decimal(100), Decimal("0.0006")))
    assert "fail_count" not in b["baseline"] and "grade_ratio" not in b["baseline"]
    assert b["baseline"]["reference"] is True


def test_write_public(tmp_path):
    lg, d = make(tmp_path)
    b = build_scoreboard(lg, d, MIDS)
    p = write_public(b, tmp_path / "repo")
    assert p == tmp_path / "repo" / "league" / "public" / "scoreboard.json"
    s = json.loads(p.read_text())
    assert s["simulated"] is True and "trades" not in s and "reasons" not in s
    rows = list(csv.DictReader((p.parent / "trades.csv").open()))
    assert rows and {"team", "ts_ms", "side", "price", "qty", "fill_grade"} <= set(rows[0])
    rs = [json.loads(x) for x in (p.parent / "reasons.jsonl").read_text().splitlines()]
    assert len(rs) == 3 and {"team", "decided_utc", "status", "reason"} <= set(rs[0])


def test_corrupt_decision_files_are_skipped(tmp_path):
    lg, d = make(tmp_path)
    (d / "claude" / "20261119T0200.json").write_text("{not json")
    (d / "gemini" / "20261119T0400.json").write_text('{"team": "gemini"}')     # 형태가 틀림
    (d / "gemini" / "20261119T0800.json").write_text(json.dumps({"team": "gemini", "decided_utc": "20261119T0800",
                                                                 "status": "ok", "items": ["x"]}))
    b = build_scoreboard(lg, d, MIDS)
    assert next(r for r in b["teams"] if r["team"] == "claude")["fail_count"] == 1
    assert len([r for r in b["reasons"] if r["team"] == "claude"]) == 2

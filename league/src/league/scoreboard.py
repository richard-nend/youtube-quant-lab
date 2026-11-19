"""기준선(BTC 현물 1배 보유)과 성적판. 순위 = 순자산, 청산 팀은 맨 아래. 공개 파일은 요약뿐(1초 원자료 없음)."""
from __future__ import annotations

import csv
import json
import re
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from .core import League
from .venues import CFG

START = Decimal(5000)
TAKER = Decimal(CFG["taker"])
ENGINE_SOURCE = re.search(r"커밋 ([0-9a-f]{40})", (Path(__file__).parent / "fill" / "SOURCE.md").read_text()).group(1)


def baseline_equity(mid_t: Decimal, mid_0: Decimal, taker_fee: Decimal) -> Decimal:
    """BTC 현물 1배 보유: 시작 자본 × 가격비 × (1 − 매수 수수료 1회). 펀딩 없음."""
    return START * mid_t / mid_0 * (1 - taker_fee)


def _decisions(decisions_dir: Path, team: str) -> list[dict]:
    return [json.loads(f.read_text()) for f in sorted((Path(decisions_dir) / team).glob("*.json"))]


def _reason(d: dict) -> str:
    return "; ".join(it["reason"] for it in d["items"] if it.get("reason")) or d.get("error", "")


def build_scoreboard(league: League, decisions_dir: Path, mids: dict) -> dict:
    """mids = {"start": BTC 중간가(시작), "now": BTC 중간가(지금)}. 반환의 trades/reasons는 write_public이 따로 쓴다."""
    rows, trades, reasons = [], [], []
    for team in league.teams:
        eng = league.engines[team]
        r, ds = eng.result(), _decisions(decisions_dir, team)
        eq, n = eng.equity(), sum(eng.grade_count.values())
        rows.append({
            "team": team, "equity": str(eq), "net_return": str(eq / START - 1),
            "max_dd": r.metrics["max_drawdown_pct"],
            "grade_ratio": {g: eng.grade_count[g] / n for g in "ABC" if eng.grade_count[g]} if n else {},
            "fail_count": sum(d["status"] == "hold_fallback" for d in ds),
            "liquidated": r.meta["flags"].get("liquidation", 0) > 0,
            "last_reason": _reason(ds[-1]) if ds else "",
        })
        side = {o["ord_id"]: o for o in r.orders}
        trades += [{"team": team, "ts_ms": f["ts_ms"], "instrument": side[f["ord_id"]]["instrument_id"],
                    "side": side[f["ord_id"]]["side"], "price": f["price"], "qty": f["qty"], "fee": f["fee"],
                    "liquidity": f["liquidity"], "fill_grade": f["fill_grade"]} for f in r.fills]
        reasons += [{"team": team, "decided_utc": d["decided_utc"], "status": d["status"], "reason": _reason(d)} for d in ds]
    rows.sort(key=lambda x: (x["liquidated"], -Decimal(x["equity"])))
    be = baseline_equity(Decimal(mids["now"]), Decimal(mids["start"]), TAKER)
    return {
        "simulated": True, "engine_source": ENGINE_SOURCE,
        "as_of_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "teams": rows,
        "baseline": {"team": "BTC 현물 1배 보유", "reference": True, "equity": str(be), "net_return": str(be / START - 1)},
        "trades": trades, "reasons": reasons,
    }


def write_public(board: dict, repo_dir: Path) -> Path:
    out = Path(repo_dir) / "league" / "public"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "scoreboard.json"
    path.write_text(json.dumps({k: v for k, v in board.items() if k not in ("trades", "reasons")}, ensure_ascii=False, indent=1))
    with open(out / "trades.csv", "w", newline="") as f:
        w = csv.DictWriter(f, ["team", "ts_ms", "instrument", "side", "price", "qty", "fee", "liquidity", "fill_grade"])
        w.writeheader()
        w.writerows(board["trades"])
    (out / "reasons.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in board["reasons"]))
    return path

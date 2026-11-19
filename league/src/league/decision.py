"""AI 판단 파일 → 엔진 주문. 검증, 크기(순자산 비율), 레버리지 상한, 중복 적용 방지."""
from __future__ import annotations

from decimal import ROUND_DOWN, Decimal
from pathlib import Path

from .fill import OrderRequest
from .fill.engine import Cancel, EngineView
from .fill.orders import Trigger
from .venues import BTC, ETH

IID = {"BTC": BTC, "ETH": ETH}
CAP = Decimal(300)   # 명목 합계 상한(%) = 순자산 3배
STEP = Decimal("0.001")  # shortcut: 거래소 lot step 설정이 없어 BTC/ETH 공통 0.001, 실제 규격 확인되면 config로


def validate(d) -> list[str]:
    if not isinstance(d, dict):
        return ["not an object"]
    errs = [f"missing {k}" for k in ("team", "decided_utc", "status", "items") if k not in d]
    if d.get("status") not in ("ok", "hold_fallback"):
        errs.append("bad status")
    if not isinstance(d.get("items"), list):
        return errs + ["items must be a list"]
    for i, it in enumerate(d["items"]):
        p = f"items[{i}]"
        if not isinstance(it, dict):
            errs.append(f"{p} not an object")
            continue
        if it.get("symbol") not in IID:
            errs.append(f"{p}.symbol")
        if it.get("action") not in ("long", "short", "close", "hold"):
            errs.append(f"{p}.action")
        n = it.get("notional_pct")
        if isinstance(n, bool) or not isinstance(n, (int, float)) or not 0 <= n <= 300:
            errs.append(f"{p}.notional_pct")
        if it.get("order") not in ("market", "limit"):
            errs.append(f"{p}.order")
        if it.get("order") == "limit" and not _pos(it.get("limit_price")):
            errs.append(f"{p}.limit_price")
        for k in ("stop", "take_profit"):
            if k in it and not _pos(it[k]):
                errs.append(f"{p}.{k}")
        if not isinstance(it.get("reason"), str):
            errs.append(f"{p}.reason")
    return errs


def _pos(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and x > 0


def to_orders(d: dict, view: EngineView, mark: dict[str, Decimal]) -> list:
    """mark 키는 "BTC"/"ETH". 판단 한 건 = 그 팀의 이전 미체결 주문 취소 + 목표 포지션까지의 차이 주문."""
    errs = validate(d)
    if errs:
        raise ValueError("; ".join(errs))
    team, tag = d["team"], f"{d['team']}:{d['decided_utc']}"
    items = [] if d["status"] == "hold_fallback" else [it for it in d["items"] if it["action"] != "hold"]
    target = {it["symbol"]: Decimal(str(it["notional_pct"])) * {"long": 1, "short": -1, "close": 0}[it["action"]]
              for it in items}
    gross = sum(abs(v) for v in target.values())
    if gross > CAP:
        target = {k: v * CAP / gross for k, v in target.items()}
    out: list = []
    eq = view.equity()
    for it in items:
        sym, iid = it["symbol"], IID[it["symbol"]]
        out += [Cancel(ord_id=o.ord_id) for o in view.open_orders(iid) if (o.req.tag or "").startswith(team + ":")]
        want = (eq * target[sym] / 100 / Decimal(mark[sym])).quantize(STEP, ROUND_DOWN) if target[sym] else Decimal(0)
        cur = view.position(iid)
        px = Decimal(str(it["limit_price"])) if it["order"] == "limit" else None
        kw = dict(tag=tag, **({"price": px, "tif": "GTC"} if px else {}))
        typ = it["order"]
        if cur and (want == 0 or (cur > 0) != (want > 0)):   # 청산 또는 방향 전환: 먼저 전량 청산
            out.append(OrderRequest(iid, "sell" if cur > 0 else "buy", "market", abs(cur), reduce_only=True, tag=tag))
            cur = Decimal(0)
        delta = want - cur
        if delta:
            out.append(OrderRequest(iid, "buy" if delta > 0 else "sell", typ, abs(delta),
                                    reduce_only=bool(cur) and (delta > 0) != (cur > 0), **kw))
        if want:
            ex = "sell" if want > 0 else "buy"
            for key, typ2 in (("stop", "stop_market"), ("take_profit", "take_profit_market")):
                if key in it:
                    out.append(OrderRequest(iid, ex, typ2, abs(want), trigger=Trigger(Decimal(str(it[key])), "mark"),
                                            reduce_only=True, tag=tag))
    return out


def applied_ids(path: Path) -> set[str]:
    path = Path(path)
    return set(path.read_text().split()) if path.exists() else set()


def mark_applied(path: Path, id: str) -> None:
    with open(path, "a") as f:
        f.write(id + "\n")

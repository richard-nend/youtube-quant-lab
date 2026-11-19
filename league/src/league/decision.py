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


def open_qty(view: EngineView, pending=()) -> tuple[dict, dict]:
    """종목별 (포지션 + 살아 있는 진입 주문 잔량, 아직 엔진에 안 들어간 진입 주문 수량). reduce-only는 노출을 늘리지 않아 뺀다.
    3배 상한은 체결될 수 있는 진입 주문까지 세야 지켜진다(리뷰 C1)."""
    live = {iid: abs(view.position(iid)) for iid in IID.values()}
    for o in view.open_orders():
        if not o.req.reduce_only and o.req.instrument_id in live:
            live[o.req.instrument_id] += o.req.qty - o.filled
    queued = dict.fromkeys(IID.values(), Decimal(0))
    for r in pending:
        if isinstance(r, OrderRequest) and not r.reduce_only and r.instrument_id in queued:
            queued[r.instrument_id] += r.qty
    return live, queued


def to_orders(d: dict, view: EngineView, mark: dict[str, Decimal], pending=(), legs: dict | None = None) -> list:
    """mark 키는 "BTC"/"ETH". 판단 한 건 = 그 팀의 이전 미체결 주문 취소 + 목표 포지션까지의 차이 주문.
    pending = 그 팀 큐에 아직 남은 요청(취소할 수 없으니 상한에 센다).
    지정가 진입의 손절·익절은 주문에 넣지 않고 legs[종목]에 담는다(체결 뒤에 데몬이 낸다: 체결 전에 닿으면 엔진이
    무포지션 reduce-only 로 취소해 익절을 잃는다, 리뷰 I3)."""
    if validate(d):   # 깨진 판단은 보유(주문·취소 없음)
        return []
    team, tag = d["team"], f"{d['team']}:{d['decided_utc']}"
    ok = lambda s: Decimal(str(mark.get(s) or 0)) > 0   # 마크 없음/0 → 그 종목은 보유
    items = [] if d["status"] == "hold_fallback" else [
        it for it in d["items"] if it["action"] != "hold" and ok(it["symbol"])]
    sign = {"long": 1, "short": -1, "close": 0}
    target = {it["symbol"]: Decimal(str(it["notional_pct"])) * sign[it["action"]] for it in items}
    flat = {s for s, v in target.items() if v == 0}
    eq = view.equity()
    if eq <= 0:
        return []
    pct = lambda s: view.position(IID[s]) * Decimal(str(mark.get(s) or 0)) / eq * 100   # 현재 부호 있는 노출(%)
    live, queued = open_qty(view, pending)
    npct = lambda s, q: q[IID[s]] * Decimal(str(mark.get(s) or 0)) / eq * 100
    # 손대지 않는 종목: 포지션 + 걸린 진입 주문. 손대는 종목의 걸린 주문은 이 판단이 취소한다. 큐의 주문은 어느 종목이든 센다.
    untouched = sum(npct(s, live) for s in IID if s not in target) + sum(npct(s, queued) for s in IID)
    gross = sum(abs(v) for v in target.values())
    if gross > max(Decimal(0), CAP - untouched):
        # 상한이 걸리면: 이미 같은 방향으로 가진 노출은 (요청 한도까지) 유지만 하고 사용분으로 센다.
        held = {k: (v / abs(v)) * min(abs(pct(k)), abs(v)) for k, v in target.items() if v and pct(k) * v > 0}
        free = {k: abs(v) for k, v in target.items() if k not in held}
        room = max(Decimal(0), CAP - untouched - sum(abs(h) for h in held.values()))
        f = min(Decimal(1), room / sum(free.values())) if free else Decimal(0)
        target = {k: held[k] if k in held else target[k] * f for k in target}
    items = [it for it in items if it["symbol"] in flat or _qty(eq, target[it["symbol"]], mark[it["symbol"]])]  # 반올림 0 → 그 종목은 보유(취소도 안 함)
    out: list = []
    for it in items:
        sym, iid = it["symbol"], IID[it["symbol"]]
        out += [Cancel(ord_id=o.ord_id) for o in view.open_orders(iid) if (o.req.tag or "").startswith(team + ":")]
        want = _qty(eq, target[sym], mark[sym])
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
        if want and typ == "limit" and delta and (delta > 0) == (want > 0):   # 지정가로 늘리는 진입: 다리는 체결 뒤
            if legs is not None and ("stop" in it or "take_profit" in it):
                legs[sym] = {"iid": iid, "exit_side": "sell" if want > 0 else "buy", "qty": str(abs(want)), "tag": tag,
                             "stop": str(it["stop"]) if "stop" in it else None,
                             "target": str(it["take_profit"]) if "take_profit" in it else None}
        elif want:
            ex = "sell" if want > 0 else "buy"
            for key, typ2 in (("stop", "stop_market"), ("take_profit", "take_profit_market")):
                if key in it:
                    out.append(OrderRequest(iid, ex, typ2, abs(want), trigger=Trigger(Decimal(str(it[key])), "mark"),
                                            reduce_only=True, tag=tag))
    return out


def _qty(eq, pct, mark) -> Decimal:
    return (eq * pct / 100 / Decimal(str(mark))).quantize(STEP, ROUND_DOWN) if pct else Decimal(0)


def applied_ids(path: Path) -> set[str]:
    path = Path(path)
    return set(path.read_text().split()) if path.exists() else set()


def mark_applied(path: Path, id: str) -> None:
    with open(path, "a") as f:
        f.write(id + "\n")

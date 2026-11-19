"""상시 데몬: 수집(Toobit·바이낸스) → 원본 저장 → Toobit 이벤트를 리그 엔진에 → 매분 판단 적용·계좌 파일
→ 매시 우리 팀 신호·스냅샷 → 하루 1회 성적판.

    uv run --project league python -m league.daemon --data <dir> [--publish]
"""
from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import logging
import signal
import subprocess
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from .collect.base import CollectorSpec
from .collect.binance import VENUE as BINANCE, BinanceUsdmCollector
from .collect.models import BookSnapshot, MarketCtx, Trade
from .collect.store import RawStore
from .collect.toobit import ToobitUsdtmCollector
from .collect.util import now_ms
from .core import TEAMS, League
from .decision import IID, applied_ids, mark_applied, open_qty, to_orders
from .events import to_engine_event
from .head import git_head
from .fill import OrderRequest, Trigger
from .fill.engine import Cancel
from .fill.events import sort_events
from .memguard import enough_memory
from .ours.sizing import size_qty
from .scoreboard import build_scoreboard, write_public
from .venues import BTC, TOOBIT

log = logging.getLogger("league.daemon")

H, DAY = 3_600_000, 86_400_000
LAG_MS = 3000             # 거래소 스트림끼리 시각이 조금 어긋나도 정렬해서 엔진에 넣는 대기
OURS_DELAY_MS = 60_000    # 정각 봉이 바이낸스 REST에 확정되도록 1분 뒤에 계산(창은 정각 기준 그대로)
LATE_MS = 10 * 60_000
AI_TEAMS = ("claude", "gpt", "gemini")
SYM = {v: k for k, v in IID.items()}
OURS_IID = {"BTCUSDT": IID["BTC"], "ETHUSDT": IID["ETH"]}
REPO = Path(__file__).resolve().parents[3]
REPLAY_KINDS = {"trades": Trade, "book": BookSnapshot, "ctx": MarketCtx}
STALE_DECISION_MS = 30 * 60_000   # 이보다 오래된 판단은 적용하지 않는다(옛 시세로 낸 판단)
SEASON = json.loads((Path(__file__).resolve().parents[2] / "config" / "season.json").read_text())


def season_start_ms() -> int:
    return int(datetime.strptime(SEASON["season_start_utc"], "%Y-%m-%dT%H:%MZ").replace(tzinfo=timezone.utc).timestamp() * 1000)


def _utc(ms: int, fmt: str) -> str:
    return datetime.fromtimestamp(ms / 1000, timezone.utc).strftime(fmt)


def _write_json(path: Path, obj) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1, default=str))
    tmp.replace(path)


def sim_exits(candidate: str) -> tuple[int, int, int]:
    """(봉 길이 ms, limit_ttl, max_hold) — 13주 러너·백테스트와 같은 asm.CANDIDATES sim 값, 없으면 simulate 기본값."""
    from quant_lab.engine.backtest import simulate

    from .ours.signals import asm
    tf, _, _, sim = asm.CANDIDATES[candidate]
    par = inspect.signature(simulate).parameters
    return {"1h": H, "4h": 4 * H}[tf], sim.get("limit_ttl", par["limit_ttl"].default), sim.get("max_hold", par["max_hold"].default)


def _default_signals(now, week_path: Path | None = None):
    from .ours.signals import new_signals   # 무거운 import(연구 코드·pandas)는 첫 회차에서만
    return new_signals(now, week_path)


class Daemon:
    def __init__(self, root: Path, *, repo: Path = REPO, publish: bool = False, signals=None, brake=None,
                 season_start_ms: int | None = None):
        self.root, self.repo, self.publish = Path(root), Path(repo), publish
        self.season_start = season_start_ms if season_start_ms is not None else globals()["season_start_ms"]()
        self.state_dir = self.root / "state"
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.store = RawStore(self.root / "raw")
        self.snap = self.state_dir / "league.pkl"
        self.applied = self.state_dir / "applied.txt"
        self.seen = self.state_dir / "ours_seen.txt"
        self.st_path = self.state_dir / "daemon.json"
        self.st = json.loads(self.st_path.read_text()) if self.st_path.exists() else {}
        self.st.setdefault("dec_equity", {})
        # 주별 후보 고정은 state/ours_week.json 에 남는다(주 중간 재시작에도 같은 후보)
        self.signals = signals or (lambda now: _default_signals(now, self.state_dir / "ours_week.json"))
        self._brake = brake
        self.buf: list = []
        self.stats = {"records": 0, "events": 0, "late": 0}
        self.minute: int | None = None
        self.halted: str | None = None   # 스냅샷 복원 실패: 원본 수집만 하고 엔진은 멈춘다
        log.info("daemon start code_head=%s data=%s", git_head(self.repo), self.root)
        if self.snap.exists() or self.snap.with_name(self.snap.name + ".prev").exists():
            try:
                self.lg = League.restore(self.snap)
            except Exception as e:
                self.halted, self.lg = repr(e)[:500], League(TEAMS, 0)   # 자리만: 돌리지도 저장하지도 않는다
                log.critical("ENGINE HALTED: snapshot restore failed (%s). Raw collection continues; fix or move "
                             "state/league.pkl(.prev) and restart.", self.halted)
            else:
                self._replay()
        else:
            self.lg = League(TEAMS, 0)

    # ---------- 이벤트 ----------
    def on_record(self, rec) -> None:
        self.store.append(rec)
        self.stats["records"] += 1
        ev = to_engine_event(rec)
        if ev is not None and not self.halted:
            self.buf.append(ev)

    def _step(self, evs) -> None:
        for ev in sort_events(evs):
            if ev.at_ms < self.lg.last_ts:
                self.stats["late"] += 1   # 엔진 시각보다 과거: 넣을 수 없다
                continue
            self.lg.step(ev)
            self.stats["events"] += 1

    def _drain(self, upto_ms: int) -> None:
        ready = [e for e in self.buf if e.at_ms <= upto_ms]
        self.buf = [e for e in self.buf if e.at_ms > upto_ms]
        self._step(ready)

    def _replay(self) -> None:
        """스냅샷 뒤에 디스크에 쌓인 Toobit 원본을 다시 넣는다(마지막 flush 이후 것은 잃는다).
        분이 바뀔 때마다 손절·익절 다리와 시간 종료도 돌린다(살아 있을 때와 같게)."""
        recs = [cls(**r) for kind, cls in REPLAY_KINDS.items()
                for r in self.store.read(TOOBIT, kind, self.lg.last_ts + 1, 2**62)]
        n, minute = self.stats["events"], None
        for ev in sort_events([e for e in map(to_engine_event, recs) if e is not None]):
            m = ev.at_ms // 60_000
            if minute is not None and m != minute and self.st.get("season_started") is not None:
                self._job("legs", self.place_legs)
                self._job("time_exits", self.time_exits, ev.at_ms)
            minute = m
            self._step([ev])
        log.info("restored snapshot at %s, replayed %d events", self.lg.last_ts, self.stats["events"] - n)

    # ---------- 시계 ----------
    def _job(self, name: str, fn, *a):
        """예약 작업 하나의 실패가 데몬을 죽이지 않게(죽으면 launchd 재시작 → 같은 실패 반복).
        실패는 작업별로 센다(누적·연속, daemon.json → 성적판 ops)."""
        key = name.split()[0]
        try:
            r = fn(*a)
        except Exception:
            log.exception("job %s failed", name)
            f = self.st.setdefault("job_failures", {}).setdefault(key, {"total": 0, "streak": 0})
            f["total"] += 1
            f["streak"] += 1
            f["last_utc"] = _utc(now_ms(), "%Y-%m-%dT%H:%M:%SZ")
            if f["streak"] in (3, 60, 3600):
                log.error("job %s failed %d times in a row", key, f["streak"])
            return None
        f = self.st.get("job_failures", {}).get(key)
        if f and f["streak"]:
            f["streak"] = 0
        return r

    def tick(self, now: int) -> None:
        if self.halted:
            m = now // 60_000
            if m != self.minute:
                self.minute = m
                self._job("flush", self.store.flush)
                log.error("ENGINE HALTED (%s): raw records=%d stored, engine not stepped", self.halted, self.stats["records"])
            return
        self._job("drain", self._drain, now - LAG_MS)
        started = self._job("season", self._season, now)
        if started:
            self._job("legs", self.place_legs)
        m = now // 60_000
        if m != self.minute:
            self.minute = m
            self._minute(now, started)
        day = _utc(now, "%Y-%m-%d")
        if self.st.get("board_day") != day:
            prev, self.st["board_day"] = self.st.get("board_day"), day   # 실패해도 날짜는 넘긴다(재시작 때 반복 안 함)
            if prev is not None:
                self._job("scoreboard", self.scoreboard)
            self._save()

    def _season(self, now: int) -> bool:
        """시즌 시작 전 = 시험 기간(수집·엔진만). 시작 시각을 처음 지날 때 한 번: 엔진을 팀당 5,000으로 새로 만들고,
        그 순간 Toobit BTC 중간가를 기준선 시작값으로, 시험 기간 판단·적용·우리 팀 기록은 trial/<시각>/ 으로 옮긴다."""
        if self.st.get("season_started") is not None:
            return True
        if now < self.season_start:
            return False
        mid = self._mid()
        if mid is None:
            log.warning("season start waits for a Toobit BTC book")
            return False
        trial = self.root / "trial" / _utc(now, "%Y%m%dT%H%M")
        trial.mkdir(parents=True, exist_ok=True)
        for p in (self.root / "decisions", self.applied, self.seen):
            if p.exists():
                p.replace(trial / p.name)
        self.lg = League(TEAMS, self.lg.last_ts)
        self.st.update(season_started=now, start_mid=str(mid), dec_equity={}, ours_open={}, ai_legs={})
        self._save()
        self.snapshot()
        log.warning("SEASON STARTED at %s: engines reset to 5000/team, start_mid %s, trial records -> %s",
                    _utc(now, "%Y-%m-%dT%H:%M:%SZ"), mid, trial)
        return True

    def _minute(self, now: int, started: bool = True) -> None:
        try:
            self.store.flush()
        except Exception:
            log.exception("raw flush failed (rows kept in memory)")
        jobs = [("decisions", self.apply_decisions, (now,)), ("time_exits", self.time_exits, (now,))] if started else []
        for name, fn, a in jobs + [("accounts", self.write_accounts, (now,)), ("results", self.update_results, ())]:
            self._job(name, fn, *a)
        log.info("minute %s records=%d events=%d late=%d buffered=%d", _utc(now, "%H:%M"), self.stats["records"],
                 self.stats["events"], self.stats["late"], len(self.buf))

    # ---------- AI 판단 ----------
    def _marks(self, team: str) -> dict:
        eng = self.lg.engines[team]
        return {s: eng.mark_of(iid) for s, iid in IID.items()}

    def apply_decisions(self, now: int) -> None:
        if not all(all(self._marks(t).values()) for t in AI_TEAMS):
            return   # 시세가 아직 없으면 판단을 소비하지 않고 미룬다
        done, changed = applied_ids(self.applied), False
        for team in AI_TEAMS:
            new = [f for f in sorted((self.root / "decisions" / team).glob("*.json")) if f"{team}:{f.stem}" not in done]
            for f in new[:-1]:   # 밀린 판단은 가장 새 것만 적용(옛 시세로 낸 판단을 겹쳐 넣지 않는다)
                mark_applied(self.applied, f"{team}:{f.stem}")
                log.warning("%s decision %s superseded, not applied", team, f.stem)
            if not new:
                continue
            f, did = new[-1], f"{team}:{new[-1].stem}"
            try:
                age = now - int(datetime.strptime(f.stem, "%Y%m%dT%H%M").replace(tzinfo=timezone.utc).timestamp() * 1000)
            except ValueError:
                age = 0   # 이름이 시각이 아니면 내용으로 판단(깨진 파일은 아래에서 유지 처리)
            if age > STALE_DECISION_MS:
                mark_applied(self.applied, did)
                log.warning("%s decision %s stale (%d min old), not applied", team, f.stem, age // 60_000)
                changed = True
                continue
            legs: dict = {}
            try:
                d = json.loads(f.read_text())
                orders = to_orders(d, self.lg.view(team), self._marks(team), self.lg.queues[team].pending, legs)
            except Exception:
                log.exception("%s decision %s unreadable, treated as hold", team, f.stem)
                orders, legs = [], {}
            mark_applied(self.applied, did)   # 먼저 기록: 재시작해도 두 번 넣지 않는다
            self.lg.submit(team, orders)
            mine = self.st.setdefault("ai_legs", {}).setdefault(team, {})
            for sym, iid in IID.items():   # 이 판단이 손댄 종목(주문을 낸 종목)의 옛 다리 정보는 버린다(주문은 판단이 취소)
                if any(getattr(o, "instrument_id", None) == iid for o in orders):
                    mine.pop(sym, None)
            mine.update(legs)
            self.st["dec_equity"][did] = str(self.lg.view(team).equity())
            log.info("%s decision %s applied: %d orders", team, f.stem, len(orders))
            changed = True
        if changed:
            self._save()
            self.snapshot()

    def update_results(self) -> None:
        """최근 판단 3개의 'result': 적용 시점 순자산 → 다음 판단 적용 시점(마지막은 지금) 순자산."""
        for team in AI_TEAMS:
            files = sorted((self.root / "decisions" / team).glob("*.json"))[-3:]
            eqs = [self.st["dec_equity"].get(f"{team}:{f.stem}") for f in files]
            for i, f in enumerate(files):
                if eqs[i] is None:
                    if f"{team}:{f.stem}" not in applied_ids(self.applied):
                        continue   # 아직 적용 전
                    res = {"applied": False}
                else:
                    end = next((e for e in eqs[i + 1:] if e is not None), None) or str(self.lg.view(team).equity())
                    a, b = Decimal(eqs[i]), Decimal(end)
                    res = {"applied": True, "equity_at_apply": eqs[i], "equity_after": end,
                           "equity_change_pct": float((b / a - 1) * 100) if a else None}
                try:
                    d = json.loads(f.read_text())
                except Exception:
                    continue
                if d.get("result") != res:
                    d["result"] = res
                    _write_json(f, d)

    def write_accounts(self, now: int) -> None:
        for team in self.lg.teams:
            v, eng = self.lg.view(team), self.lg.engines[team]
            _write_json(self.state_dir / f"account_{team}.json", {
                "as_of_utc": _utc(now, "%Y-%m-%dT%H:%MZ"), "equity": v.equity(), "cash": v.cash,
                "positions": [{"symbol": s, "qty": v.position(iid), "entry": v.entry(iid), "mark": eng.mark_of(iid)}
                              for s, iid in IID.items() if v.position(iid)],
                "open_orders": [{"symbol": SYM.get(o.req.instrument_id), "side": o.req.side, "type": o.req.type,
                                 "qty": o.req.qty - o.filled, "price": o.req.price,
                                 "trigger": o.req.trigger.price if o.req.trigger else None, "reduce_only": o.req.reduce_only}
                                for o in v.open_orders()],
            })

    # ---------- 우리 팀(매시) ----------
    def due_hour(self, now: int) -> int | None:
        if self.halted:
            return None
        hour = now // H * H
        if now - hour < OURS_DELAY_MS or self.st.get("last_hour") == hour:
            return None
        if now - hour > LATE_MS:
            log.warning("ours round for %s is late (%d min after the hour)", _utc(hour, "%m-%d %H:00"), (now - hour) // 60_000)
        return hour

    def fetch_signals(self, hour: int):
        """None = 이번 회차 건너뜀(이유는 로그). 스레드에서 불러도 되게 리그 상태는 건드리지 않는다."""
        if not enough_memory():
            log.warning("ours round %s skipped: available memory < 3GB", _utc(hour, "%m-%d %H:00"))
            return None
        try:
            if self._brake is None:
                from .ours.brake import Brake
                self._brake = Brake.load()
            return self.signals(datetime.fromtimestamp(hour / 1000, timezone.utc))
        except Exception:
            log.exception("ours round %s failed", _utc(hour, "%m-%d %H:00"))
            return None

    def run_hour(self, hour: int, sigs) -> None:
        prev = self.st.get("last_hour")
        if prev is not None and hour - prev > H:
            missed = [_utc(t, "%m-%d %H:00") for t in range(prev + H, hour, H)]
            log.warning("ours: missed %d hour(s) %s — no catch-up, their signals are not entered", len(missed), missed)
        if self.st.get("season_started") is None:
            if sigs:
                log.info("ours: before season start, %d signal(s) not entered", len(sigs))
            sigs = ()
        for s in sigs or ():
            self._job(f"enter {s.key}", self._enter, s)
        self.st["last_hour"] = hour
        self._save()
        self.snapshot()

    def _enter(self, s) -> None:
        if s.key in applied_ids(self.seen):
            return
        mark_applied(self.seen, s.key)   # 먼저 기록: 재시작해도 다시 들어가지 않는다
        blocked, score = self._brake.blocks(s.facts)
        if blocked:
            log.info("ours %s blocked by brake (score %.3f)", s.key, score)
            return
        iid, v, eng = OURS_IID[s.symbol], self.lg.view("ours"), self.lg.engines["ours"]
        mine = [o for o in v.open_orders(iid) if (o.req.tag or "").startswith("ours:")]
        if v.position(iid) or any(not o.req.reduce_only for o in mine):
            log.info("ours %s skipped: %s already in a trade (no overlap, as the backtest)", s.key, s.symbol)
            return
        mark = eng.mark_of(iid)
        if not mark:
            log.warning("ours %s skipped: no Toobit mark yet", s.key)
            return
        entry = Decimal(str(s.limit_price)) if s.entry == "limit" else mark
        live, queued = open_qty(v, self.lg.queues["ours"].pending)   # 포지션 + 걸린·큐의 진입 주문(리뷰 C1)
        held = sum((live[i] + queued[i]) * (eng.mark_of(i) or 0) for i in IID.values())
        qty = size_qty(v.equity(), entry, Decimal(str(s.stop)), held)
        if not qty:
            log.info("ours %s skipped: size rounds to 0", s.key)
            return
        side, ex = ("buy", "sell") if s.side > 0 else ("sell", "buy")
        tag = f"ours:{s.key}"
        kw = {"price": entry, "tif": "GTC"} if s.entry == "limit" else {}
        # 손절·익절은 체결 뒤에 place_legs 가 낸다(무포지션 reduce-only 는 엔진이 취소; backtest 도 체결 봉부터 본다)
        self.lg.submit("ours", [Cancel(ord_id=o.ord_id) for o in mine] + [   # 지난 거래의 남은 손절·익절 정리
            OrderRequest(iid, side, s.entry, qty, tag=tag, **kw)])
        step, ttl, hold = sim_exits(s.candidate)
        sig_ms = int(s.signal_time.timestamp() * 1000)   # 신호 봉의 시작 시각(봉 index)
        self.st.setdefault("ours_open", {})[tag] = {
            "iid": iid, "step_ms": step, "max_hold": hold, "signal_ms": sig_ms,
            "exit_side": ex, "qty": str(qty), "stop": str(s.stop), "target": str(s.target),
            # simulate: 지정가는 신호 다음 봉부터 limit_ttl 개 봉 안에 닿아야 체결 → 봉 (signal+ttl) 마감에 만료
            "expire_ms": sig_ms + (ttl + 1) * step if s.entry == "limit" else None}
        self._save()
        log.info("ours %s entered: %s %s %s (brake score %.3f)", s.key, side, qty, s.symbol, score)

    def place_legs(self) -> None:
        """포지션이 있는데 살아 있는 손절/익절이 없으면 (다시) 낸다. 수량 = 진입 주문 수량(reduce-only 라 포지션까지만 체결).
        우리 팀(ours_open)과 AI 팀의 지정가 진입(ai_legs, 리뷰 I3) 모두."""
        self._legs("ours", list(self.st.get("ours_open", {}).items()))
        for team in AI_TEAMS:
            mine = self.st.get("ai_legs", {}).get(team, {})
            eng = self.lg.engines[team]
            for sym, t in list(mine.items()):   # 진입이 끝났고(체결·취소) 포지션도 없으면 거래 끝: 정보 버림
                entry = next((o for o in eng.orders if o.req.tag == t["tag"] and o.req.instrument_id == t["iid"]
                              and not o.req.reduce_only), None)
                if entry is not None and not entry.live and not self.lg.view(team).position(t["iid"]):
                    del mine[sym]
            self._legs(team, [(t["tag"], t) for t in mine.values()])

    def _legs(self, team: str, items) -> None:
        if self.lg.queues[team].pending:
            return   # 낸 주문이 아직 엔진에 안 들어감
        v, out = self.lg.view(team), []
        for tag, t in items:
            pos = v.position(t["iid"])
            if not pos or (pos > 0) != (t["exit_side"] == "sell"):
                continue
            live = {o.req.type for o in v.open_orders(t["iid"]) if o.req.tag == tag and o.req.reduce_only}
            for typ, px in (("stop_market", t.get("stop")), ("take_profit_market", t.get("target"))):
                if px is not None and typ not in live:
                    out.append(OrderRequest(t["iid"], t["exit_side"], typ, Decimal(t["qty"]),
                                            trigger=Trigger(Decimal(px), "mark"), reduce_only=True, tag=tag))
                    log.info("%s %s: %s placed after fill", team, tag, typ)
        if out:
            self.lg.submit(team, out)

    def time_exits(self, now: int) -> None:
        """backtest simulate 의 시간 규칙: 지정가 미체결 만료(limit_ttl), 보유 max_hold 봉 뒤 그 봉 마감에 시장가 청산.
        손절·익절로 이미 끝난 거래는 남은 주문을 치우고 목록에서 뺀다."""
        opened, eng, v = self.st.get("ours_open", {}), self.lg.engines["ours"], self.lg.view("ours")
        changed = False
        for tag, t in list(opened.items()):
            iid = t["iid"]
            entry = next((o for o in eng.orders if o.req.tag == tag and not o.req.reduce_only), None)
            if entry is None:
                if any(getattr(r, "tag", None) == tag for r in self.lg.queues["ours"].pending):
                    continue   # 아직 큐에 있음(다음 이벤트에 엔진으로)
                del opened[tag]   # 엔진에도 큐에도 없음: 진입 기록 뒤 스냅샷 전에 죽었다 → 추적만 남은 것
                changed = True
                log.warning("ours %s dropped: entry order neither in the engine nor queued (crash before snapshot)", tag)
                continue
            rest = [o for o in v.open_orders(iid) if o.req.tag == tag]
            out, why = [], None
            if entry.live and not entry.filled and t["expire_ms"] and now >= t["expire_ms"]:
                out, why = [Cancel(ord_id=o.ord_id) for o in rest], "limit expired"
            elif entry.filled:
                if t.get("close_ms") is None:
                    fill_ms = next(f["ts_ms"] for f in eng.fills if f["ord_id"] == entry.ord_id)
                    bar = t["signal_ms"] + (fill_ms - t["signal_ms"]) // t["step_ms"] * t["step_ms"]
                    t["close_ms"] = bar + (t["max_hold"] + 1) * t["step_ms"]   # 봉 (fill + max_hold) 마감, 종가 = 그때 시장가
                pos = v.position(iid)
                if not pos and not entry.live:
                    out, why = [Cancel(ord_id=o.ord_id) for o in rest], "closed by stop/target"
                elif now >= t["close_ms"]:
                    out = [Cancel(ord_id=o.ord_id) for o in rest]
                    if pos:
                        out.append(OrderRequest(iid, "sell" if pos > 0 else "buy", "market", abs(pos), reduce_only=True, tag=tag))
                    why = "max_hold"
            elif not entry.live:
                out, why = [Cancel(ord_id=o.ord_id) for o in rest], "entry not filled (rejected/cancelled)"
            if why:
                self.lg.submit("ours", out)
                del opened[tag]
                changed = True
                log.info("ours %s done: %s", tag, why)
        if changed:
            self.snapshot()   # 스냅샷 먼저: 그 사이 죽어도 추적이 남아 다시 판단(청산은 reduce-only라 중복 무해)
        self._save()

    # ---------- 저장·성적판 ----------
    def _mid(self) -> Decimal | None:
        b = self.lg.engines[self.lg.teams[0]].market(BTC).book
        return (b.bids[0][0] + b.asks[0][0]) / 2 if b and b.bids and b.asks else None

    def _save(self) -> None:
        _write_json(self.st_path, self.st)

    def snapshot(self) -> None:
        self.lg.snapshot(self.snap)

    def scoreboard(self) -> None:
        if self.st.get("season_started") is None:
            log.info("scoreboard skipped: season not started (nothing is published before the start)")
            return
        mid = self._mid()
        if mid is None or self.st.get("start_mid") is None:
            log.warning("scoreboard skipped: no BTC book yet")
            return
        ops = {"job_failures": self.st.get("job_failures", {})}
        path = write_public(build_scoreboard(self.lg, self.root / "decisions", {"start": self.st["start_mid"], "now": mid}, ops),
                            self.repo)
        log.info("scoreboard written %s", path)
        if self.publish:
            subprocess.Popen(["/bin/zsh", str(self.repo / "league" / "deploy" / "publish.sh")])   # 기다리지 않는다

    def close(self) -> None:
        """멈출 때: 버퍼를 디스크에(먼저), 남은 이벤트를 엔진에, 스냅샷."""
        self._job("flush", self.store.flush)
        self._save()
        if self.halted:
            log.error("stopped while ENGINE HALTED: snapshot left untouched")
            return
        self._job("drain", self._drain, 2**62)
        self.snapshot()
        log.info("stopped: records=%d events=%d late=%d", self.stats["records"], self.stats["events"], self.stats["late"])


async def run(d: Daemon, stop: asyncio.Event) -> None:
    cols = [ToobitUsdtmCollector(CollectorSpec(TOOBIT), d.on_record), BinanceUsdmCollector(CollectorSpec(BINANCE), d.on_record)]
    tasks = [asyncio.create_task(c.run(stop)) for c in cols]
    try:
        while not stop.is_set():
            now = now_ms()
            d._job("tick", d.tick, now)
            hour = d.due_hour(now)
            if hour is not None:   # 신호 계산은 스레드에서(수집은 계속), 주문은 이 루프에서
                sigs = await asyncio.to_thread(d.fetch_signals, hour)
                d._job("run_hour", d.run_hour, hour, sigs)
                d.st["last_hour"] = hour   # 실패한 회차를 매초 다시 돌지 않게
            try:
                await asyncio.wait_for(stop.wait(), 1.0)
            except TimeoutError:
                pass
    finally:
        stop.set()
        await asyncio.gather(*tasks, return_exceptions=True)
        d.close()


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="league.daemon")
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--repo", type=Path, default=REPO)
    ap.add_argument("--publish", action="store_true", help="성적판을 쓴 뒤 deploy/publish.sh 실행")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    d = Daemon(a.data, repo=a.repo, publish=a.publish)

    async def go():
        stop = asyncio.Event()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, stop.set)
        await run(d, stop)

    asyncio.run(go())


if __name__ == "__main__":
    main()

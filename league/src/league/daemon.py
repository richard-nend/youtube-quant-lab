"""상시 데몬: 수집(Toobit·바이낸스) → 원본 저장 → Toobit 이벤트를 리그 엔진에 → 매분 판단 적용·계좌 파일
→ 매시 우리 팀 신호·스냅샷 → 하루 1회 성적판.

    uv run --project league python -m league.daemon --data <dir> [--publish]
"""
from __future__ import annotations

import argparse
import asyncio
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
from .decision import IID, applied_ids, mark_applied, to_orders
from .events import to_engine_event
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


def _utc(ms: int, fmt: str) -> str:
    return datetime.fromtimestamp(ms / 1000, timezone.utc).strftime(fmt)


def _write_json(path: Path, obj) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1, default=str))
    tmp.replace(path)


def _default_signals(now):
    from .ours.signals import new_signals   # 무거운 import(연구 코드·pandas)는 첫 회차에서만
    return new_signals(now)


class Daemon:
    def __init__(self, root: Path, *, repo: Path = REPO, publish: bool = False, signals=None, brake=None):
        self.root, self.repo, self.publish = Path(root), Path(repo), publish
        self.state_dir = self.root / "state"
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.store = RawStore(self.root / "raw")
        self.snap = self.state_dir / "league.pkl"
        self.applied = self.state_dir / "applied.txt"
        self.seen = self.state_dir / "ours_seen.txt"
        self.st_path = self.state_dir / "daemon.json"
        self.st = json.loads(self.st_path.read_text()) if self.st_path.exists() else {}
        self.st.setdefault("dec_equity", {})
        self.signals, self._brake = signals or _default_signals, brake
        self.buf: list = []
        self.stats = {"records": 0, "events": 0, "late": 0}
        self.minute: int | None = None
        if self.snap.exists():
            self.lg = League.restore(self.snap)
            self._replay()
        else:
            self.lg = League(TEAMS, 0)

    # ---------- 이벤트 ----------
    def on_record(self, rec) -> None:
        self.store.append(rec)
        self.stats["records"] += 1
        ev = to_engine_event(rec)
        if ev is not None:
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
        """스냅샷 뒤에 디스크에 쌓인 Toobit 원본을 다시 넣는다(마지막 flush 이후 것은 잃는다)."""
        recs = [cls(**r) for kind, cls in REPLAY_KINDS.items()
                for r in self.store.read(TOOBIT, kind, self.lg.last_ts + 1, 2**62)]
        n = self.stats["events"]
        self._step([e for e in map(to_engine_event, recs) if e is not None])
        log.info("restored snapshot at %s, replayed %d events", self.lg.last_ts, self.stats["events"] - n)

    # ---------- 시계 ----------
    def tick(self, now: int) -> None:
        self._drain(now - LAG_MS)
        if self.st.get("start_mid") is None and self._mid():
            self.st["start_mid"] = str(self._mid())
            self._save()
        m = now // 60_000
        if m != self.minute:
            self.minute = m
            self._minute(now)
        day = _utc(now, "%Y-%m-%d")
        if self.st.get("board_day") != day:
            if self.st.get("board_day") is not None:
                self.scoreboard()
            self.st["board_day"] = day
            self._save()

    def _minute(self, now: int) -> None:
        try:
            self.store.flush()
        except Exception:
            log.exception("raw flush failed (rows kept in memory)")
        self.apply_decisions()
        self.write_accounts(now)
        self.update_results()
        log.info("minute %s records=%d events=%d late=%d buffered=%d", _utc(now, "%H:%M"), self.stats["records"],
                 self.stats["events"], self.stats["late"], len(self.buf))

    # ---------- AI 판단 ----------
    def _marks(self, team: str) -> dict:
        eng = self.lg.engines[team]
        return {s: eng.mark_of(iid) for s, iid in IID.items()}

    def apply_decisions(self) -> None:
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
                d = json.loads(f.read_text())
                orders = to_orders(d, self.lg.view(team), self._marks(team))
            except Exception:
                log.exception("%s decision %s unreadable, treated as hold", team, f.stem)
                orders = []
            mark_applied(self.applied, did)   # 먼저 기록: 재시작해도 두 번 넣지 않는다
            self.lg.submit(team, orders)
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
        for s in sigs or ():
            self._enter(s)
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
        held = sum(abs(v.position(i)) * (eng.mark_of(i) or 0) for i in IID.values())
        qty = size_qty(v.equity(), entry, Decimal(str(s.stop)), held)
        if not qty:
            log.info("ours %s skipped: size rounds to 0", s.key)
            return
        side, ex = ("buy", "sell") if s.side > 0 else ("sell", "buy")
        tag = f"ours:{s.key}"
        kw = {"price": entry, "tif": "GTC"} if s.entry == "limit" else {}
        self.lg.submit("ours", [Cancel(ord_id=o.ord_id) for o in mine] + [   # 지난 거래의 남은 손절·익절 정리
            OrderRequest(iid, side, s.entry, qty, tag=tag, **kw),
            OrderRequest(iid, ex, "stop_market", qty, trigger=Trigger(Decimal(str(s.stop)), "mark"), reduce_only=True, tag=tag),
            OrderRequest(iid, ex, "take_profit_market", qty, trigger=Trigger(Decimal(str(s.target)), "mark"),
                         reduce_only=True, tag=tag)])
        log.info("ours %s entered: %s %s %s (brake score %.3f)", s.key, side, qty, s.symbol, score)

    # ---------- 저장·성적판 ----------
    def _mid(self) -> Decimal | None:
        b = self.lg.engines[self.lg.teams[0]].market(BTC).book
        return (b.bids[0][0] + b.asks[0][0]) / 2 if b and b.bids and b.asks else None

    def _save(self) -> None:
        _write_json(self.st_path, self.st)

    def snapshot(self) -> None:
        self.lg.snapshot(self.snap)

    def scoreboard(self) -> None:
        mid = self._mid()
        if mid is None or self.st.get("start_mid") is None:
            log.warning("scoreboard skipped: no BTC book yet")
            return
        path = write_public(build_scoreboard(self.lg, self.root / "decisions", {"start": self.st["start_mid"], "now": mid}), self.repo)
        log.info("scoreboard written %s", path)
        if self.publish:
            subprocess.Popen(["/bin/zsh", str(self.repo / "league" / "deploy" / "publish.sh")])   # 기다리지 않는다

    def close(self) -> None:
        """멈출 때: 남은 이벤트를 엔진에, 버퍼를 디스크에, 스냅샷."""
        self._drain(2**62)
        try:
            self.store.flush()
        except Exception:
            log.exception("raw flush on stop failed")
        self._save()
        self.snapshot()
        log.info("stopped: records=%d events=%d late=%d", self.stats["records"], self.stats["events"], self.stats["late"])


async def run(d: Daemon, stop: asyncio.Event) -> None:
    cols = [ToobitUsdtmCollector(CollectorSpec(TOOBIT), d.on_record), BinanceUsdmCollector(CollectorSpec(BINANCE), d.on_record)]
    tasks = [asyncio.create_task(c.run(stop)) for c in cols]
    try:
        while not stop.is_set():
            now = now_ms()
            d.tick(now)
            hour = d.due_hour(now)
            if hour is not None:   # 신호 계산은 스레드에서(수집은 계속), 주문은 이 루프에서
                d.run_hour(hour, await asyncio.to_thread(d.fetch_signals, hour))
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

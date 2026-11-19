"""수집기 공통 실행 틀: 웹소켓 연결·재연결(지수 백오프)·구독·주기 조회."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
import logging
import random
import time
from typing import Awaitable, Callable, Iterable

import websockets

from .models import Record

@dataclass(frozen=True)
class CollectorSpec:
    venue: str
    symbols: tuple[str, ...] = ("BTCUSDT", "ETHUSDT")
    book_depth: int = 20
    book_interval_ms: int = 1000
    ctx_poll_seconds: float = 60.0
    instruments_poll_seconds: float = 86400.0


log = logging.getLogger(__name__)

Emit = Callable[[Record], None]
PollFn = Callable[[], Awaitable[None]]
Poller = PollFn | tuple[PollFn, float] | tuple[PollFn, float, bool]


def next_boundary_delay(now_s: float, every: float, offset: float = 5.0) -> float:
    """다음 UTC 주기 경계(+offset초)까지 남은 초. 경계 직후라 offset 안이면 그 경계를 쓴다."""
    return every - ((now_s - offset) % every)


class VenueCollector:
    name = "base"

    def __init__(self, cfg: CollectorSpec, emit: Emit):
        self.cfg = cfg
        self._emit = emit
        self.stats = {"messages": 0, "records": 0, "reconnects": 0, "errors": 0}

    def emit(self, recs: Iterable[Record]) -> None:
        for r in recs:
            self._emit(r)
            self.stats["records"] += 1

    # 하위 클래스가 구현
    def ws_url(self) -> str: raise NotImplementedError
    def ws_urls(self) -> list[str]: return [self.ws_url()]  # 연결이 여러 개 필요한 거래소는 재정의
    async def on_open(self, ws) -> None: ...
    def handle(self, msg: dict) -> list[Record]: raise NotImplementedError
    # 주기 조회: 함수, (함수, 주기초) 또는 (함수, 주기초, 정렬). 함수만 주면 poll_seconds() 주기.
    # 정렬=True: 시작 때 한 번 + UTC 주기 경계마다(86400이면 매일 00:00 UTC). 시작 시각 기준으로 돌면
    # 일별 저장이 하루 끝에야 생기고, 재시작이 겹치면 날짜 하나를 건너뛸 수 있다(10-26 instruments).
    def pollers(self) -> list[Poller]: return []
    async def keepalive(self, ws) -> None: ...

    async def run(self, stop: asyncio.Event) -> None:
        tasks = [asyncio.create_task(self._ws_loop(url, stop)) for url in self.ws_urls()]
        for p in self.pollers():
            fn, every, align = (p + (False,))[:3] if isinstance(p, tuple) else (p, self.poll_seconds(), False)
            tasks.append(asyncio.create_task(self._poll_loop(fn, every, stop, align)))
        await stop.wait()
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _ws_loop(self, url: str, stop: asyncio.Event) -> None:
        backoff = 1.0
        while not stop.is_set():
            try:
                async with websockets.connect(url, ping_interval=20, ping_timeout=20,
                                              max_size=2**24) as ws:
                    await self.on_open(ws)
                    ka = asyncio.create_task(self.keepalive(ws))
                    backoff = 1.0
                    try:
                        async for raw in ws:
                            if raw == "pong":   # 문자열 핑에 문자열로 답하는 거래소(Bitget)
                                continue
                            self.stats["messages"] += 1
                            try:
                                self.emit(self.handle(json.loads(raw)))
                            except Exception:  # 메시지 하나의 오류로 연결을 끊지 않는다
                                self.stats["errors"] += 1
                                log.exception("%s: handle error", self.name)
                    finally:
                        ka.cancel()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                self.stats["reconnects"] += 1
                log.warning("%s: ws error %r, reconnect in %.1fs", self.name, e, backoff)
                await asyncio.sleep(backoff + random.random())
                backoff = min(backoff * 2, 60.0)

    async def _poll_loop(self, fn: PollFn, every: float, stop: asyncio.Event, align: bool = False) -> None:
        loop = asyncio.get_running_loop()
        while not stop.is_set():
            started = loop.time()
            try:
                await fn()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                self.stats["errors"] += 1
                log.warning("%s: poll error %r", self.name, e)
            if align:
                await asyncio.sleep(next_boundary_delay(time.time(), every))
            else:
                # 조회에 걸린 시간을 빼서 주기를 지킨다 (호가 2초 간격 등)
                await asyncio.sleep(max(0.0, every - (loop.time() - started)))

    def poll_seconds(self) -> float:
        return self.cfg.ctx_poll_seconds

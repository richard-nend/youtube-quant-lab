"""Toobit USDT-M 무기한 수집기 (적색·내부 전용).

약관 §6.4·§16.1(재배포·파생·상업 이용 제한) → 바이낸스와 같은 방식: 서버 내부 백테스트·합성 데이터 계산 전용.
근거: docs/research/2026-10-26-toobit.md

엔드포인트(공식 Agent Kit toobit-docs/agent-kit + 2026-10-26 EC2 실측):
- 웹소켓 wss://stream.toobit.com/quote/ws/v1, 구독 {"symbol","topic","event":"sub","params":{"binary":false}}
  - trade: data [{"v"(id),"t","p","q","m"}], m=true 면 매수자가 메이커 → 매도 공격
  - depth: data [{"s","t","b":[[p,q]],"a":[[p,q]]}] (스냅샷)
  - markPrice 토픽은 빈 데이터만 와서 REST로 조회
- REST https://api.toobit.com: /quote/v1/markPrice, /api/v1/futures/fundingRate(period "8H"),
  /quote/v1/openInterest, /quote/v1/index?symbol=BTCUSDT, /api/v1/exchangeInfo(contracts)
- 단위: 체결·호가 수량은 **계약 수**(BTC 1계약 = contractMultiplier 0.001 BTC) → 코인으로 환산해 저장.
  1분봉 거래량×가격 ≈ 거래대금으로 확인. openInterest의 size는 코인 단위로 판단(BTC 약 3만 개 ≈ 24시간
  거래대금의 0.67배; 계약 단위면 30 BTC로 비현실적) — 공식 문서로 재확인 필요.
"""
from __future__ import annotations

import asyncio
import json

import httpx

from . import util as ids
from .base import CollectorSpec, VenueCollector
from .models import BookSnapshot, Instrument, MarketCtx, Record, Trade
from .util import Throttle, f, now_ms

WS_URL = "wss://stream.toobit.com/quote/ws/v1"
REST = "https://api.toobit.com"
VENUE = "toobit_usdtm"


def _ident(symbol: str) -> dict:
    return {"instrument_id": ids.toobit_swap(symbol), "market_type": ids.PERP}


def parse_trades(symbol: str, data: list[dict], recv: int, mult: float) -> list[Record]:
    return [Trade(ts_ms=int(t["t"]), venue=VENUE, symbol=symbol, price=float(t["p"]),
                  size=float(t["q"]) * mult, side="sell" if t.get("m") else "buy",
                  trade_id=str(t["v"]), recv_ms=recv, **_ident(symbol))
            for t in data]


def parse_depth(symbol: str, d: dict, recv: int, depth: int, mult: float) -> BookSnapshot:
    return BookSnapshot(ts_ms=int(d["t"]), venue=VENUE, symbol=symbol,
                        bids=[[float(p), float(q) * mult] for p, q in d["b"][:depth]],
                        asks=[[float(p), float(q) * mult] for p, q in d["a"][:depth]],
                        recv_ms=recv, **_ident(symbol))


def parse_period_h(period: str | None) -> float | None:
    """'8H' → 8.0"""
    if period and period.upper().endswith("H"):
        return f(period[:-1])
    return None


def parse_ctx(symbol: str, mark: dict, funding: dict, oi: dict, index: dict, recv: int) -> MarketCtx:
    index_token = symbol.replace("-SWAP-", "")
    oi_row = next((r for r in oi.get("openInterestList", []) if r.get("symbol") == symbol), {})
    nf = funding.get("nextFundingTime")
    return MarketCtx(
        ts_ms=int(mark.get("time") or recv), venue=VENUE, symbol=symbol, recv_ms=recv,
        funding_rate=f(funding.get("rate")), mark_price=f(mark.get("price")),
        index_price=f(index.get("index", {}).get(index_token)), open_interest=f(oi_row.get("size")),
        next_funding_ms=int(nf) if nf else None, funding_interval_h=parse_period_h(funding.get("period")),
        **_ident(symbol),
    )


def parse_contracts(info: dict, recv: int) -> list[Record]:
    out: list[Record] = []
    for c in info.get("contracts", []):
        try:
            iid = ids.toobit_swap(c["symbol"])
        except ValueError:
            continue
        filt = {x["filterType"]: x for x in c.get("filters", [])}
        tiers = c.get("riskLimits")
        out.append(Instrument(
            ts_ms=recv, venue=VENUE, symbol=c["symbol"], recv_ms=recv, asset=ids.asset_of(c["underlying"]),
            quote=c["quoteAsset"], tick_size=f(filt.get("PRICE_FILTER", {}).get("tickSize")),
            lot_size=f(filt.get("LOT_SIZE", {}).get("stepSize")),
            min_notional=f(filt.get("MIN_NOTIONAL", {}).get("minNotional")),
            max_leverage=max((f(t.get("maxLeverage")) or 0 for t in tiers), default=None) if tiers else None,
            margin_tiers=json.dumps(tiers) if tiers else None,
            contract_multiplier=f(c.get("contractMultiplier")), status=(c.get("status") or "").lower() or None,
            instrument_id=iid, market_type=ids.PERP,
        ))
    return out


class ToobitUsdtmCollector(VenueCollector):
    name = VENUE

    def __init__(self, cfg: CollectorSpec, emit):
        super().__init__(cfg, emit)
        self.depth = cfg.book_depth
        self.book_throttle = Throttle(cfg.book_interval_ms)
        # 리그 설정은 'BTCUSDT' 형식 → Toobit 'BTC-SWAP-USDT' 로 바꾼다(이미 swap 형식이면 그대로)
        self.syms = [s if "-" in s else "{}-SWAP-{}".format(*ids.split_concat(s)) for s in cfg.symbols]
        for s in self.syms:
            ids.toobit_swap(s)  # 형식이 틀리면 시작할 때 바로 실패
        self.mult: dict[str, float] = {}

    def ws_url(self) -> str:
        return WS_URL

    async def _get(self, c: httpx.AsyncClient, path: str, **params):
        r = await c.get(path, params=params)
        r.raise_for_status()
        return r.json()

    async def load_multipliers(self) -> None:
        async with httpx.AsyncClient(base_url=REST, timeout=30) as c:
            info = await self._get(c, "/api/v1/exchangeInfo")
        self.mult = {i.symbol: i.contract_multiplier for i in parse_contracts(info, now_ms())
                     if i.contract_multiplier}

    async def on_open(self, ws) -> None:
        await self.load_multipliers()  # 환산값 없이 저장하지 않는다: 실패하면 예외 → 재연결 대기
        missing = [s for s in self.syms if s not in self.mult]
        if missing:
            raise RuntimeError(f"toobit: no contractMultiplier for {missing}")
        for s in self.syms:
            for topic in ("trade", "depth"):
                await ws.send(json.dumps({"symbol": s, "topic": topic, "event": "sub",
                                          "params": {"binary": False}}))

    async def keepalive(self, ws) -> None:
        while True:
            await asyncio.sleep(30)
            await ws.send(json.dumps({"ping": now_ms()}))

    def handle(self, msg: dict) -> list[Record]:
        topic, sym, data, recv = msg.get("topic"), msg.get("symbol"), msg.get("data"), now_ms()
        if not data or sym not in self.mult:
            return []  # pong, 오류 응답 등
        if topic == "trade":
            return parse_trades(sym, data, recv, self.mult[sym])
        if topic == "depth":
            snap = parse_depth(sym, data[0], recv, self.depth, self.mult[sym])
            return [snap] if self.book_throttle.allow(sym, snap.ts_ms) else []
        return []

    def poll_seconds(self) -> float:
        return self.cfg.ctx_poll_seconds

    def pollers(self):
        async def poll_ctx():
            async with httpx.AsyncClient(base_url=REST, timeout=10) as c:
                recs = []
                for s in self.syms:
                    mark = await self._get(c, "/quote/v1/markPrice", symbol=s)
                    funding = (await self._get(c, "/api/v1/futures/fundingRate", symbol=s) or [{}])[0]
                    oi = await self._get(c, "/quote/v1/openInterest", symbol=s)
                    index = await self._get(c, "/quote/v1/index", symbol=s.replace("-SWAP-", ""))
                    recs.append(parse_ctx(s, mark, funding, oi, index, now_ms()))
                self.emit(recs)

        async def poll_instruments():
            async with httpx.AsyncClient(base_url=REST, timeout=30) as c:
                self.emit(parse_contracts(await self._get(c, "/api/v1/exchangeInfo"), now_ms()))

        every = self.cfg.instruments_poll_seconds
        return [poll_ctx, (poll_instruments, every, True)]

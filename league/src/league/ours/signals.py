"""우리 팀 신호: 13주 러너와 같은 바이낸스 마감 봉·월별 후보 선택, EP08b와 같은 사전 사실(facts).

연구 코드는 import만 한다(파일 경로로 importlib 로드, 러너와 같은 방식). 데이터(바이낸스 아카이브 캐시, EP08b rows)는
gitignore된 본 체크아웃에 있으므로 QL_MAIN_ROOT에서 읽는다.
"""
from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from quant_lab import LAB_ROOT
from quant_lab.data import binance
from quant_lab.engine.backtest import simulate

MAIN_ROOT = Path(os.environ.get("QL_MAIN_ROOT", "/Users/gimhyeong-gi/MyProjects/Youtube/trading/quant_lab"))
# shortcut: quant_lab의 캐시 경로(LAB_ROOT/.cache)를 이 프로세스에서만 본 체크아웃 캐시로 돌린다; quant_lab에 경로 설정이 생기면 그걸로
binance.CACHE_DIR = MAIN_ROOT / ".cache"
SYMBOLS = ("BTCUSDT", "ETHUSDT")


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, LAB_ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


asm = _load("assembly", "episodes/ep07/assembly_walkforward.py")
mb = _load("meta_label_b", "episodes/ep08/meta_label_b.py")
runner = _load("live_runner", "live/runner.py")  # import 부작용은 live/start.json 읽기뿐; main()은 돌지 않는다


@dataclass(frozen=True)
class LiveSignal:
    key: str
    symbol: str
    candidate: str
    signal_time: pd.Timestamp
    side: int
    entry: str
    limit_price: float | None
    stop: float
    target: float
    facts: dict


def chosen_candidate(trades, month):
    """live/runner.py chosen_candidate 그대로: 직전 12개월 순수익 합 최대, 30건 미만이면 제외."""
    lo = month - pd.DateOffset(months=12)
    score = {n: (float(asm.month_slice(t, lo, month).net_ret.sum()) if len(asm.month_slice(t, lo, month)) >= 30 else -np.inf)
             for n, t in trades.items()}
    best = max(score, key=score.get)
    return None if score[best] == -np.inf else best


def run_candidate(b: pd.DataFrame, candidate: str):
    """(signals, trades) of one candidate on these bars, exactly as the runner and build_trades do."""
    _, zwin, fn, sim = asm.CANDIDATES[candidate]
    sigs = fn(b) if fn else asm.flow_signals(b, zwin)
    return sigs, simulate(b, sigs, asm.COSTS, **sim)


def live_facts(sym, candidate, signal_time, bars, trades, sigs=None) -> dict:
    """EP08b build_trades의 x와 같은 키·값. bars·trades는 signal_time까지의 정보만 담겨 있어야 한다."""
    tf, zwin, fn, sim = asm.CANDIDATES[candidate]
    b = bars
    if sigs is None:
        sigs = fn(b) if fn else asm.flow_signals(b, zwin)
    s = next(x for x in sigs if b.index[x.bar] == signal_time)  # build_trades: first signal at that time
    k = b.index.get_loc(signal_time)
    hours = 1 if tf == "1h" else 4
    ref = s.limit_price if s.entry == "limit" else float(b.close.iloc[k])
    stop_pct = 100 * abs(ref - s.stop) / ref
    target_pct = 100 * abs(s.target - ref) / ref
    cost_pct = 100 * mb.exante_cost(s.entry, asm.COSTS)
    m = signal_time.to_period("M").to_timestamp().tz_localize("UTC")
    w = asm.month_slice(trades, m - pd.DateOffset(months=12), m)
    wr, mean_net, _ = mb.recent_record(trades, signal_time)
    f = mb._context(b, asm).loc[signal_time]
    return {"long": float(s.side > 0), "limit_entry": float(s.entry == "limit"),
            "stop_pct": stop_pct, "target_pct": target_pct, "reward_risk": target_pct / stop_pct,
            "cost_pct": cost_pct, "cost_to_stop": cost_pct / stop_pct, "max_hold_h": sim.get("max_hold", 96) * hours,
            "trail12_sum_pct": 100 * float(w.net_ret.sum()), "trail12_n": float(len(w)),
            "recent_win": wr, "recent_mean_pct": None if mean_net is None else 100 * mean_net,
            "hour": float(signal_time.hour), "weekday": float(signal_time.weekday()),
            **{k2: (None if pd.isna(v) else float(v)) for k2, v in f.items()},
            **{f"cand_{c}": float(c == candidate) for c in asm.CANDIDATES}}


def new_signals(now: datetime) -> list[LiveSignal]:
    """이번 시간(now-1h, now]에 봉이 마감되며 생긴 신호 중, 그 달 선택 후보의 것만."""
    out = []
    for sym in SYMBOLS:
        bb = {tf: runner.bars(sym, tf, now) for tf in ("1h", "4h")}
        runs = {c: run_candidate(bb[tf], c) for c, (tf, *_) in asm.CANDIDATES.items()}
        trades = {c: t for c, (_, t) in runs.items()}
        for cand, (tf, *_) in asm.CANDIDATES.items():
            b, (sigs, t) = bb[tf], runs[cand]
            step = b.index[1] - b.index[0]
            for s in sigs:
                ts = b.index[s.bar]
                key = f"{sym}|{cand}|{ts.isoformat()}"
                if not (now - timedelta(hours=1) < ts + step <= now) or any(x.key == key for x in out):
                    continue  # one signal per bar, the first (as build_trades)
                if chosen_candidate(trades, ts.to_period("M").to_timestamp().tz_localize("UTC")) != cand:
                    continue
                out.append(LiveSignal(key, sym, cand, ts, int(s.side), s.entry,
                                      s.limit_price, float(s.stop), float(s.target),
                                      live_facts(sym, cand, ts, b, t, sigs)))
    return out

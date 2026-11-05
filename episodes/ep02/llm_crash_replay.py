"""EP02 — replay every ~$68k crash event minute by minute with an LLM trader.

Pre-registered design (fixed BEFORE looking at any LLM output):
  data      BTCUSDT USDT-M perp 1m bars; events from crash_event_finder.py (all of them)
  window    decisions every STEP_MIN minutes from trigger-4m for HORIZON_MIN minutes
  input     last 60 one-minute OHLCV bars, prices indexed to 100 at the first shown bar,
            relative minute labels only (no dates, no symbol) + RSI(14) + position state
  account   $10,000, cross margin, max leverage 20x, taker 0.05% + 1.5bp slippage per fill
  fills     decision on bar t's close -> filled at bar t+1 open
  stops     optional stop_pct from the model, checked on every 1m bar (gaps fill at open)
  liquid.   when equity <= 0.4% maintenance margin of notional -> account wiped (equity 0)
  arms      single  : one LLM call per step
            debate  : bull call + bear call + risk-manager call (decides) per step
            rsi_rule: no LLM. flat & RSI<35 -> long 25% of max notional; add 25% each further
                      -0.5% while RSI<35 (max 3 adds); close when RSI>55; no stop
  scoring   end-of-window mark-to-market return, liquidation flag, # of adds, latency

Usage:
  uv run python episodes/ep02/llm_crash_replay.py --pilot          # 1 event, 1 run per arm
  uv run python episodes/ep02/llm_crash_replay.py --runs 5 --debate-runs 2
"""

from __future__ import annotations

import argparse
import json
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from quant_lab import LAB_ROOT, RESULTS_DIR
from quant_lab.data import binance
from quant_lab.engine.costs import CostModel
from quant_lab.llm import Gemini
from quant_lab.results import write_result

MODEL = "gemini-3.8-flash"
STEP_MIN = 2
HORIZON_MIN = 40
PRE_MIN = 4
HISTORY = 60
EQUITY0 = 10_000.0
LEVERAGE = 20
MMR = 0.004
COSTS = CostModel()
LOG_DIR = LAB_ROOT / "runs" / "ep02"

DECISION_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "action": {"type": "STRING", "enum": ["open_long", "open_short", "add", "hold", "close"]},
        "size_frac": {"type": "NUMBER", "description": "fraction (0-1) of max notional (equity x 20) for open/add"},
        "stop_pct": {"type": "NUMBER", "description": "stop distance in % from average entry; 0 = no stop"},
        "reason": {"type": "STRING"},
    },
    "required": ["action", "size_frac", "stop_pct", "reason"],
}
ARGUMENT_SCHEMA = {
    "type": "OBJECT",
    "properties": {"argument": {"type": "STRING"}, "recommended_action": {"type": "STRING",
                   "enum": ["open_long", "open_short", "add", "hold", "close"]}},
    "required": ["argument", "recommended_action"],
}

SYSTEM_TRADER = (
    "You are a discretionary crypto perpetual-futures day trader managing a $10,000 account with up to "
    "20x leverage. Every 2 minutes you receive the latest one-minute candles and your position. "
    "Prices are indexed (first shown bar = 100). Decide what to do now. "
    "size_frac is the fraction of your maximum notional (equity x 20) to use for open/add. "
    "stop_pct is the stop distance in percent from your average entry (0 means no stop). "
    "Answer only with the JSON object."
)
SYSTEM_BULL = ("You are the BULL researcher on a trading desk. Using only the data given, make the strongest "
               "case for being long right now and recommend an action. Be concise (max 80 words).")
SYSTEM_BEAR = ("You are the BEAR researcher on a trading desk. Using only the data given, make the strongest "
               "case against being long right now, attacking weaknesses in the bullish view, and recommend "
               "an action. Be concise (max 80 words).")
SYSTEM_RISK = (SYSTEM_TRADER + " You are the final risk manager: you read the bull and bear researchers' "
               "arguments and make the decision, enforcing sound risk management.")


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)


@dataclass
class Account:
    equity: float = EQUITY0          # realised cash balance
    side: int = 0
    qty: float = 0.0                 # BTC
    avg: float = 0.0
    stop: float | None = None
    adds: int = 0
    fills: list = field(default_factory=list)
    liquidated: bool = False
    fees: float = 0.0

    def notional(self, px):
        return self.qty * px

    def upnl(self, px):
        return self.side * self.qty * (px - self.avg)

    def mtm(self, px):
        return self.equity + self.upnl(px)

    def fill(self, t, px, side, notional_usd, kind):
        px_eff = px * (1 + side * COSTS.slippage_bps / 10_000)
        fee = notional_usd * COSTS.taker_fee
        q = notional_usd / px_eff
        if self.side == 0:
            self.side, self.qty, self.avg = side, q, px_eff
        else:
            self.avg = (self.avg * self.qty + px_eff * q) / (self.qty + q)
            self.qty += q
        self.equity -= fee
        self.fees += fee
        self.fills.append({"t": str(t), "kind": kind, "px": px_eff, "notional": notional_usd})

    def close(self, t, px, kind):
        if self.side == 0:
            return
        px_eff = px * (1 - self.side * COSTS.slippage_bps / 10_000)
        fee = self.qty * px_eff * COSTS.taker_fee
        self.equity += self.side * self.qty * (px_eff - self.avg) - fee
        self.fees += fee
        self.fills.append({"t": str(t), "kind": kind, "px": px_eff, "notional": self.qty * px_eff})
        self.side, self.qty, self.avg, self.stop = 0, 0.0, 0.0, None


def process_bar(acc: Account, t, bar) -> None:
    """Stops and liquidation inside one 1m bar (adverse extreme checked first)."""
    if acc.side == 0 or acc.liquidated:
        return
    worst = bar.low if acc.side > 0 else bar.high
    if acc.stop is not None:
        hit = worst <= acc.stop if acc.side > 0 else worst >= acc.stop
        if hit:
            gap = (bar.open < acc.stop) if acc.side > 0 else (bar.open > acc.stop)
            acc.close(t, bar.open if gap else acc.stop, "stop")
            return
    if acc.mtm(worst) <= MMR * acc.notional(worst):
        acc.fills.append({"t": str(t), "kind": "LIQUIDATION", "px": float(worst)})
        acc.equity, acc.side, acc.qty, acc.avg, acc.stop, acc.liquidated = 0.0, 0, 0.0, 0.0, None, True


def apply_decision(acc: Account, t, open_px, d: dict) -> None:
    a = d.get("action", "hold")
    frac = float(np.clip(d.get("size_frac", 0) or 0, 0, 1))
    max_notional = acc.mtm(open_px) * LEVERAGE
    room = max(max_notional - acc.notional(open_px), 0)
    if a in ("open_long", "open_short"):
        side = 1 if a == "open_long" else -1
        if acc.side == -side:
            acc.close(t, open_px, "reverse")
            room = acc.mtm(open_px) * LEVERAGE
        if acc.side == 0 and frac > 0:
            acc.fill(t, open_px, side, min(frac * max_notional, room), a)
    elif a == "add" and acc.side != 0 and frac > 0 and room > 0:
        acc.fill(t, open_px, acc.side, min(frac * max_notional, room), "add")
        acc.adds += 1
    elif a == "close":
        acc.close(t, open_px, "close")
    sp = float(d.get("stop_pct", 0) or 0)
    if acc.side != 0:
        acc.stop = acc.avg * (1 - acc.side * sp / 100) if sp > 0 else None


def market_prompt(hist: pd.DataFrame, base: float, acc: Account, px: float, r: float) -> str:
    lines = ["minute,open,high,low,close,volume_rel"]
    vmean = hist["volume"].mean()
    n = len(hist)
    for i, (_, b) in enumerate(hist.iterrows()):
        lines.append(f"t-{n-1-i},{b.open/base*100:.3f},{b.high/base*100:.3f},{b.low/base*100:.3f},"
                     f"{b.close/base*100:.3f},{b.volume/vmean:.2f}")
    if acc.side == 0:
        pos = "FLAT"
    else:
        liq_move = (acc.mtm(px) - MMR * acc.notional(px)) / acc.qty
        liq_px = px - acc.side * liq_move
        pos = (f"{'LONG' if acc.side > 0 else 'SHORT'} notional ${acc.notional(px):,.0f} "
               f"({acc.notional(px)/acc.mtm(px):.1f}x), avg entry {acc.avg/base*100:.3f}, "
               f"unrealised PnL ${acc.upnl(px):,.0f}, stop {('%.3f' % (acc.stop/base*100)) if acc.stop else 'none'}, "
               f"est. liquidation {liq_px/base*100:.3f}, adds so far {acc.adds}")
    return ("\n".join(lines) + f"\n\nRSI(14, 1m) = {r:.1f}\nEquity ${acc.mtm(px):,.0f}\nPosition: {pos}\n")


def run_one(event: dict, bars: pd.DataFrame, arm: str, run_id: int, llm: Gemini | None) -> dict:
    trig = pd.Timestamp(event["trigger_time"])
    start = trig - pd.Timedelta(minutes=PRE_MIN)
    end = start + pd.Timedelta(minutes=HORIZON_MIN)
    r_series = rsi(bars["close"])
    acc = Account()
    latencies, steps = [], []
    t = start
    while t < end and not acc.liquidated:
        hist = bars.loc[:t].iloc[-HISTORY:]
        base = hist["open"].iloc[0]
        px = float(bars.loc[t, "close"])
        r = float(r_series.loc[t])
        prompt = market_prompt(hist, base, acc, px, r)
        tag = f"{event['trigger_time']}|{arm}|{run_id}|{t}"
        if arm == "rsi_rule":
            d = rule_decision(acc, px, r)
            lat = 0.0
        elif arm == "single":
            out = llm.json(SYSTEM_TRADER, prompt, DECISION_SCHEMA, tag=tag)
            d, lat = out["response"], out["latency_s"]
        else:  # debate
            bull = llm.json(SYSTEM_BULL, prompt, ARGUMENT_SCHEMA, tag=tag + "|bull")
            bear = llm.json(SYSTEM_BEAR, prompt, ARGUMENT_SCHEMA, tag=tag + "|bear")
            rm_prompt = (prompt + f"\nBULL: {json.dumps(bull['response'])}\nBEAR: {json.dumps(bear['response'])}\n")
            fin = llm.json(SYSTEM_RISK, rm_prompt, DECISION_SCHEMA, tag=tag + "|risk")
            d = fin["response"]
            lat = bull["latency_s"] + bear["latency_s"] + fin["latency_s"]  # sequential pipeline
        latencies.append(lat)
        steps.append({"t": str(t), "price_idx": px / base * 100, "rsi": r, "decision": d, "latency_s": lat,
                      "equity_mtm": acc.mtm(px)})
        # execute at next bar open, then walk 1m bars until the next decision
        nxt = t + pd.Timedelta(minutes=1)
        if nxt in bars.index:
            apply_decision(acc, nxt, float(bars.loc[nxt, "open"]), d)
        for tt in pd.date_range(nxt, t + pd.Timedelta(minutes=STEP_MIN), freq="1min", inclusive="left"):
            if tt in bars.index:
                process_bar(acc, tt, bars.loc[tt])
        t = t + pd.Timedelta(minutes=STEP_MIN)

    final_px = float(bars.loc[:end, "close"].iloc[-1])
    final = acc.mtm(final_px)
    eq_path = [s["equity_mtm"] for s in steps] + [final]
    return {
        "event": event["trigger_time"], "arm": arm, "run": run_id,
        "return_pct": final / EQUITY0 - 1,
        "min_equity_pct": min(eq_path) / EQUITY0 - 1,
        "liquidated": acc.liquidated,
        "adds": acc.adds,
        "traded": len(acc.fills) > 0,
        "first_action_long_after_trigger": next((f["kind"] for f in acc.fills), None),
        "fees": acc.fees,
        "latency_median_s": statistics.median(latencies) if latencies else None,
        "latency_max_s": max(latencies) if latencies else None,
        "fills": acc.fills,
        "steps": steps,
    }


def rule_decision(acc: Account, px: float, r: float) -> dict:
    if acc.side == 0 and r < 35:
        return {"action": "open_long", "size_frac": 0.25, "stop_pct": 0, "reason": "RSI<35"}
    if acc.side > 0 and r > 55:
        return {"action": "close", "size_frac": 0, "stop_pct": 0, "reason": "RSI>55"}
    if acc.side > 0 and r < 35 and acc.adds < 3 and acc.fills and px <= acc.fills[-1]["px"] * 0.995:
        return {"action": "add", "size_frac": 0.25, "stop_pct": 0, "reason": "RSI<35, -0.5% lower"}
    return {"action": "hold", "size_frac": 0, "stop_pct": 0, "reason": "rule"}


def summarise(rows: list[dict]) -> dict:
    out = {}
    for arm in sorted({r["arm"] for r in rows}):
        rs = [r for r in rows if r["arm"] == arm]
        rets = np.array([r["return_pct"] for r in rs])
        out[arm] = {
            "runs": len(rs),
            "mean_return": float(rets.mean()), "median_return": float(np.median(rets)),
            "worst_return": float(rets.min()), "best_return": float(rets.max()),
            "share_losing": float((rets < 0).mean()),
            "liquidations": int(sum(r["liquidated"] for r in rs)),
            "runs_with_3plus_adds": int(sum(r["adds"] >= 3 for r in rs)),
            "share_traded": float(np.mean([r["traded"] for r in rs])),
            "latency_median_s": (float(np.median([r["latency_median_s"] for r in rs if r["latency_median_s"]]))
                                 if arm != "rsi_rule" else 0.0),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--debate-runs", type=int, default=2)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    events = json.loads((RESULTS_DIR / "ep02.json").read_text())["experiments"]["crash_events"]["values"]["events"]
    if args.pilot:
        events = events[:1]
        args.runs, args.debate_runs = 1, 1
    frames = []
    for e in events:
        t = pd.Timestamp(e["trigger_time"])
        frames.append(binance.klines("BTCUSDT", "1m", (t - pd.Timedelta(days=1)).date(),
                                     (t + pd.Timedelta(days=1)).date(), market="um"))
    bars = pd.concat(frames)
    bars = bars[~bars.index.duplicated()].sort_index()

    tag = "pilot" if args.pilot else "full"
    llm = Gemini(MODEL, log_path=LOG_DIR / f"calls_{tag}.jsonl")
    jobs = []
    for e in events:
        jobs.append((e, "rsi_rule", 0))
        jobs += [(e, "single", i) for i in range(args.runs)]
        jobs += [(e, "debate", i) for i in range(args.debate_runs)]

    # resumable: every finished run is appended immediately; reruns skip finished jobs
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    done_path = LOG_DIR / f"rows_{tag}.jsonl"
    rows = [json.loads(l) for l in done_path.read_text().splitlines()] if done_path.exists() else []
    done = {(r["event"], r["arm"], r["run"]) for r in rows}
    jobs = [j for j in jobs if (j[0]["trigger_time"], j[1], j[2]) not in done]
    print(f"{len(rows)} runs already done, {len(jobs)} to go", flush=True)
    with ThreadPoolExecutor(args.workers) as ex:
        futs = {ex.submit(run_one, e, bars, arm, i, None if arm == "rsi_rule" else llm): (e, arm, i)
                for e, arm, i in jobs}
        for f in as_completed(futs):
            try:
                r = f.result()
            except Exception as exc:  # one failed run must not kill the batch
                e, arm, i = futs[f]
                print(f"FAILED {e['trigger_time']} {arm} #{i}: {str(exc)[:200]}", flush=True)
                continue
            rows.append(r)
            with done_path.open("a") as fh:
                fh.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
            print(f"{r['event'][:16]} {r['arm']:8} #{r['run']} ret {r['return_pct']:+.1%} "
                  f"min {r['min_equity_pct']:+.1%} liq={r['liquidated']} adds={r['adds']} "
                  f"lat={r['latency_median_s']}", flush=True)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    (LOG_DIR / f"runs_{tag}.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1, default=str))
    summary = summarise(rows)
    print(json.dumps(summary, indent=1))
    if not args.pilot:
        write_result("ep02", "llm_crash_replay", {"summary": summary,
                     "per_run": [{k: v for k, v in r.items() if k not in ("steps", "fills")} for r in rows]},
                     params={"model": MODEL, "backend": llm.backend, "step_min": STEP_MIN, "horizon_min": HORIZON_MIN, "leverage": LEVERAGE,
                             "equity0": EQUITY0, "costs": COSTS.as_dict(), "runs": args.runs,
                             "debate_runs": args.debate_runs, "n_events": len(events)},
                     note=f"full prompts/responses: quant_lab/runs/ep02/calls_{tag}.jsonl")


if __name__ == "__main__":
    main()

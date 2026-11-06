"""Runs INSIDE the legacy ai-hedge-fund venv (invoked by ai_hedge_fund_probe.py --pipeline).

Uses the built-in src.backtesting.BacktestEngine unchanged. The only wrapper is the agent callable:
the full hedge-fund graph (all 13 personas + 6 analytic agents + risk manager + portfolio manager)
runs on the FIRST trading day of each month; on other days the wrapper returns "hold" for every
ticker without calling any LLM (budget: daily decisions would need ~11,000 LLM calls).

argv: out_dir start_date end_date call_cap
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path

OUT = Path(sys.argv[1])
START, END, CAP = sys.argv[2], sys.argv[3], int(sys.argv[4])
TICKERS = ["AAPL", "GOOGL", "MSFT", "NVDA", "TSLA"]
OUT.mkdir(parents=True, exist_ok=True)

import src.utils.llm as U  # noqa: E402  (patch before agents import call_llm)

_lock = threading.Lock()
_tl = threading.local()
STATE = {"calls": 0, "errors": 0, "fallbacks": [], "date": None}
CALL_LOG = (OUT / "llm_calls.jsonl").open("a")


def _txt(p):
    return p.to_string() if hasattr(p, "to_string") else str(p)


class LoggedLLM:
    def __init__(self, llm):
        self._llm = llm

    def invoke(self, prompt):
        with _lock:
            STATE["calls"] += 1
            n = STATE["calls"]
            if n > CAP:
                raise SystemExit(f"LLM call cap {CAP} exceeded")
        t0 = time.perf_counter()
        err = content = None
        try:
            r = self._llm.invoke(prompt)
            content = r.content if isinstance(r.content, str) else json.dumps(r.content)
            return r
        except Exception as e:
            err = f"{type(e).__name__}: {e}"[:500]
            with _lock:
                STATE["errors"] += 1
            raise
        finally:
            with _lock:
                CALL_LOG.write(json.dumps({"n": n, "date": STATE["date"], "agent": getattr(_tl, "agent", None),
                                           "latency_s": round(time.perf_counter() - t0, 3), "error": err,
                                           "prompt": _txt(prompt), "response": content}, ensure_ascii=False) + "\n")
                CALL_LOG.flush()

    def __getattr__(self, k):
        return getattr(self._llm, k)


_orig_get_model, _orig_call = U.get_model, U.call_llm
U.get_model = lambda *a, **k: LoggedLLM(_orig_get_model(*a, **k))


def call_llm(prompt, pydantic_model, agent_name=None, state=None, max_retries=3, default_factory=None):
    _tl.agent = agent_name
    fell = []

    def df():
        txt = _txt(prompt)
        fell.append(1)
        with _lock:
            STATE["fallbacks"].append({"date": STATE["date"], "agent": agent_name,
                                       "tickers_in_prompt": [t for t in TICKERS if t in txt]})
        return default_factory()

    return _orig_call(prompt, pydantic_model, agent_name, state, max_retries, df if default_factory else None)


U.call_llm = call_llm

from src.main import run_hedge_fund  # noqa: E402
from src.backtesting.engine import BacktestEngine  # noqa: E402

for k in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):  # load_dotenv may pull them in; Vertex/ADC only
    os.environ.pop(k, None)
assert os.environ.get("VERTEX_PROJECT"), "VERTEX_PROJECT must be set"

RUNS = []
_seen_months: set[str] = set()


def monthly_agent(*, tickers, start_date, end_date, portfolio, model_name, model_provider, selected_analysts):
    month = end_date[:7]
    if month in _seen_months:
        return {"decisions": {t: {"action": "hold", "quantity": 0} for t in tickers}, "analyst_signals": {}}
    _seen_months.add(month)
    STATE["date"] = end_date
    t0 = time.time()
    out = run_hedge_fund(tickers=tickers, start_date=start_date, end_date=end_date, portfolio=portfolio,
                         show_reasoning=False, selected_analysts=selected_analysts,
                         model_name=model_name, model_provider=model_provider)
    RUNS.append({"date": end_date, "lookback_start": start_date, "portfolio_in": portfolio,
                 "decisions": out["decisions"], "analyst_signals": out["analyst_signals"],
                 "wall_s": round(time.time() - t0, 1)})
    (OUT / "rebalances.json").write_text(json.dumps(RUNS, indent=1, default=str))
    print(f"[runner] {end_date} done, LLM calls so far {STATE['calls']}", file=sys.stderr, flush=True)
    return out


engine = BacktestEngine(agent=monthly_agent, tickers=TICKERS, start_date=START, end_date=END,
                        initial_capital=100_000.0, model_name="gemini-3.8-flash", model_provider="Google",
                        selected_analysts=None, initial_margin_requirement=0.0)

TRADES = []
_orig_exec = engine._executor.execute_trade


def logged_exec(ticker, action, quantity, current_price, portfolio):
    q = _orig_exec(ticker, action, quantity, current_price, portfolio)
    if q:
        TRADES.append({"date": STATE.get("today"), "ticker": ticker, "action": action, "qty": q, "price": current_price})
    return q


engine._executor.execute_trade = logged_exec

# record the engine's current date for trade logging
_orig_run_agent = engine._agent_controller.run_agent


def run_agent(agent, **kw):
    STATE["today"] = kw["end_date"]
    return _orig_run_agent(agent, **kw)


engine._agent_controller.run_agent = run_agent

metrics = engine.run_backtest()
pv = [{**{k: (str(v) if k == "Date" else v) for k, v in p.items()}} for p in engine.get_portfolio_values()]
(OUT / "portfolio_values.json").write_text(json.dumps(pv, indent=1, default=str))
(OUT / "trades.json").write_text(json.dumps(TRADES, indent=1))
(OUT / "summary.json").write_text(json.dumps({"engine_metrics": metrics, "llm_calls": STATE["calls"],
                                              "llm_errors": STATE["errors"], "fallbacks": STATE["fallbacks"]},
                                             indent=1, default=str))
print(f"[runner] finished. LLM calls {STATE['calls']}, errors {STATE['errors']}, fallbacks {len(STATE['fallbacks'])}",
      file=sys.stderr)

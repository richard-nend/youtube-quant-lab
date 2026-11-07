"""EP05 — reproducible probe of virattt/ai-hedge-fund (legacy persona version).

Pre-registered design (fixed BEFORE any LLM output was seen):

Code under test
  repo      virattt/ai-hedge-fund @ 6c41ae8cf5fb4e30eb8cdb6a816eb56ff234989d (2026-07-31, last commit
            before the 2026-08-02 "Make 2.0.0 the default" restructure a7a99e5). Checked out at
            vendor/ai-hedge-fund-legacy, own poetry venv (.venv), Python 3.12.
  patch     src/llm/models.py, Google branch: if VERTEX_PROJECT is set, return
            langchain_google_vertexai.ChatVertexAI(model, project, location=global) (ADC creds,
            bills GCP). langchain-google-vertexai<3 added to the legacy venv. No other change.
  model     gemini-3.8-flash on Vertex AI (project from VERTEX_PROJECT). No AI Studio key.

Pipeline experiment (a)-(c)  [requires financialdatasets.ai data]
  universe  AAPL, GOOGL, MSFT, NVDA, TSLA
  dates     first NYSE trading day of each month Mar..Oct 2025 (8 dates)
  run       all 13 personas + 4 analytic agents -> risk manager -> portfolio manager, per date
  (a) agreement   for each persona pair: share of (ticker,date) cells with identical signal;
                  signal correlation = Pearson corr of s*conf, s in {bull:+1, neutral:0, bear:-1}
  (b) cancel rate share of (ticker,date) cells where PM action is hold, and where
                  |mean(s*conf/100)| over personas < 0.10 ("near-zero net signal")
  (c) backtest    built-in src/backtesting engine Mar-03-2025..Oct-31-2025, $100k, vs equal-weight
                  buy-and-hold of the 5 tickers; turnover = sum|trade notional| / mean equity
  STATUS (first attempt): blocked. financialdatasets.ai returns HTTP 402/401 for every endpoint
          without a key (checked 2026-11-06, incl. AAPL). No key/account created, nothing paid.
          `--check-data` re-runs that check.
  STATUS (owner-approved second attempt): DATA LAYER SUBSTITUTED -> the agents saw substitute data.
          vendor/ai-hedge-fund-legacy/DATA_SUBSTITUTIONS.md has the full table. In short:
            prices        Yahoo chart API daily OHLCV (split-adj., not dividend-adj.)
            fundamentals  SEC EDGAR XBRL companyfacts, only facts with filed <= analysis date;
                          ratios computed by our own formulas; TTM = last 4 fiscal quarters
            insider       SEC EDGAR Form 4 XML, filing date <= analysis date
            news          NONE (empty list) -> Munger, Burry, Taleb, Lynch, Fisher, Druckenmiller lose
                          news input; News-Sentiment agent is always neutral/0; Sentiment agent is
                          insider-only
          Agents, prompts, risk manager, portfolio manager and backtest engine are unchanged.
  cadence the built-in BacktestEngine runs daily from 2025-03-03 to 2025-10-31 (marks, fills at
          that day's close); the agent graph runs on the first trading day of each month (8 dates)
          and returns hold on other days (daily would need ~11k LLM calls). Signals for (a)/(b)
          come from those 8 graph runs (8 dates x 5 tickers = 40 cells, 13 personas each).
  extra (descriptive, not pre-registered): same (a)/(b) stats over all 18 non-risk agents,
          LLM fallback count (call_llm default_factory -> neutral), per-persona bull/bear/neutral mix.
  caveat  the contamination probe shows gemini-3.8-flash recalls 12-month returns for windows ending
          up to mid-2025 almost exactly; Mar-Oct 2025 may be partly memorised, so (c) is not a clean
          out-of-sample test.

Contamination probe (d)  [no market data given to the model]
  question  "Estimate <ticker>'s price return over the 12 months after <date>" (JSON: return_pct,
            knows: bool). No tools, no data in the prompt. temperature 0.
  universe  same 5 tickers
  dates     first trading day of January and July, 2018..2025 (16 dates) -> 80 calls,
            + 1 call asking the model for its own training cutoff (recorded, not used for buckets)
  truth     Yahoo Finance chart API daily adjclose: close on the date -> close on the last trading
            day <= date + 365 calendar days
  buckets   by window END date:  past   = end <= 2024-06-30
                                 recent = 2024-07-01 .. 2025-06-30
                                 future = end >= 2025-07-01
  metrics   per bucket: sign accuracy, MAE (pp), Spearman rank corr(pred, actual),
            share with |pred - actual| <= 10pp, share knows=true;
            baseline: "always +" sign accuracy (megacaps mostly rose) and the error of always
            predicting the in-bucket mean actual return
  claim     contamination supported iff past-bucket accuracy is clearly better than future-bucket
            (MAE lower, rank corr higher) — memorised outcomes, not forecasting skill.

Usage (from quant_lab/):
  uv run python episodes/ep05/ai_hedge_fund_probe.py --check-data
  uv run python episodes/ep05/ai_hedge_fund_probe.py --contamination
  uv run python episodes/ep05/ai_hedge_fund_probe.py --pipeline        # legacy graph + engine, then --analyze
  uv run python episodes/ep05/ai_hedge_fund_probe.py --analyze runs/ep05/persona_backtest_2025-03-03_2025-10-31
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from quant_lab import LAB_ROOT
from quant_lab.llm import Gemini
from quant_lab.results import write_result

MODEL = "gemini-3.8-flash"
TICKERS = ["AAPL", "GOOGL", "MSFT", "NVDA", "TSLA"]
LEGACY_COMMIT = "6c41ae8cf5fb4e30eb8cdb6a816eb56ff234989d"
RESTRUCTURE_COMMIT = "a7a99e5b4cbbe789f4e16bd02d794e86a2a3a3d2"
LOG_DIR = LAB_ROOT / "runs" / "ep05"
YAHOO = "https://query1.finance.yahoo.com/v8/finance/chart/{t}?period1={p1}&period2={p2}&interval=1d&events=split,div"
FDS = "https://api.financialdatasets.ai"

SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "return_pct": {"type": "NUMBER", "description": "price return in percent, e.g. 23.5 or -12"},
        "knows": {"type": "BOOLEAN", "description": "true if you recall the actual outcome, false if guessing"},
    },
    "required": ["return_pct", "knows"],
}
CUTOFF_SCHEMA = {"type": "OBJECT", "properties": {"cutoff": {"type": "STRING"}}, "required": ["cutoff"]}
SYSTEM = "You are a financial historian. Answer from memory only. You have no tools and no data feed."


def check_data() -> dict:
    out = {}
    for path in ["prices/?ticker=AAPL&interval=day&interval_multiplier=1&start_date=2025-03-01&end_date=2025-03-10",
                 "financial-metrics/?ticker=AAPL&report_period_lte=2025-03-03&limit=1&period=ttm",
                 "news/?ticker=AAPL&end_date=2025-03-03&limit=1",
                 "insider-trades/?ticker=AAPL&filing_date_lte=2025-03-03&limit=1",
                 "company/facts/?ticker=AAPL"]:
        r = requests.get(f"{FDS}/{path}", timeout=30)  # deliberately no X-API-KEY
        out[path.split("/?")[0]] = r.status_code
    return out


def yahoo_adjclose(ticker: str) -> pd.Series:
    p1 = int(datetime(2017, 12, 1, tzinfo=timezone.utc).timestamp())
    p2 = int(datetime.now(timezone.utc).timestamp())
    r = requests.get(YAHOO.format(t=ticker, p1=p1, p2=p2), headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    r.raise_for_status()
    res = r.json()["chart"]["result"][0]
    idx = pd.to_datetime(res["timestamp"], unit="s", utc=True).tz_convert("America/New_York").normalize().tz_localize(None)
    return pd.Series(res["indicators"]["adjclose"][0]["adjclose"], index=idx).dropna()


def probe_dates(px: pd.Series) -> list[pd.Timestamp]:
    out = []
    for y in range(2018, 2026):
        for m in (1, 7):
            out.append(px.index[px.index >= pd.Timestamp(y, m, 1)][0])
    return out


def bucket(end: pd.Timestamp) -> str:
    if end <= pd.Timestamp(2024, 6, 30):
        return "past"
    if end <= pd.Timestamp(2025, 6, 30):
        return "recent"
    return "future"


def summarise(df: pd.DataFrame) -> dict:
    out = {}
    for b in ["past", "recent", "future", "all"]:
        d = df if b == "all" else df[df.bucket == b]
        if d.empty:
            continue
        err = (d.pred - d.actual).abs()
        out[b] = {
            "n": len(d),
            "sign_accuracy": float((np.sign(d.pred) == np.sign(d.actual)).mean()),
            "always_up_sign_accuracy": float((d.actual > 0).mean()),
            "mae_pp": float(err.mean()),
            "median_ae_pp": float(err.median()),
            "baseline_mae_pp_predict_bucket_mean": float((d.actual - d.actual.mean()).abs().mean()),
            "within_10pp": float((err <= 10).mean()),
            "spearman": float(d.pred.rank().corr(d.actual.rank())),
            "knows_true_share": float(d.knows.mean()),
            "mean_actual_pct": float(d.actual.mean()),
            "mean_pred_pct": float(d.pred.mean()),
        }
    return out


def contamination() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    llm = Gemini(MODEL, log_path=LOG_DIR / f"contamination_calls_{stamp}.jsonl", temperature=0.0, backend="vertex")
    cutoff = llm.json(SYSTEM, "What is your training data cutoff (month and year)? Answer briefly.",
                      CUTOFF_SCHEMA, tag="cutoff")["response"]["cutoff"]

    jobs = []
    for t in TICKERS:
        px = yahoo_adjclose(t)
        for d0 in probe_dates(px):
            d1 = px.index[px.index <= d0 + timedelta(days=365)][-1]
            actual = (px[d1] / px[d0] - 1) * 100
            jobs.append({"ticker": t, "date": d0.date().isoformat(), "end": d1.date().isoformat(),
                         "actual": float(actual), "bucket": bucket(d1)})

    def ask(j):
        q = (f"What was {j['ticker']}'s stock price return (split/dividend-adjusted, in percent) over the 12 months "
             f"after {j['date']}, i.e. from the close on {j['date']} to the close about one year later? "
             f"Give your single best number even if unsure, and say whether you actually recall it.")
        r = llm.json(SYSTEM, q, SCHEMA, tag=f"{j['ticker']}_{j['date']}")["response"]
        return {**j, "pred": float(r["return_pct"]), "knows": bool(r["knows"])}

    with ThreadPoolExecutor(max_workers=8) as ex:
        rows = list(ex.map(ask, jobs))
    df = pd.DataFrame(rows)
    df.to_csv(LOG_DIR / f"contamination_rows_{stamp}.csv", index=False)
    summary = summarise(df)
    print(json.dumps({"self_reported_cutoff": cutoff, **summary}, indent=2))

    fds = check_data()
    write_result(
        "ep05", "ai_hedge_fund_probe",
        {
            "pipeline_experiment": {
                "status": "blocked",
                "reason": "financialdatasets.ai requires a paid key / x402 payment for all endpoints incl. AAPL",
                "http_status_without_key": fds,
            },
            "contamination": {
                "self_reported_cutoff": cutoff,
                "by_window_end_bucket": summary,
                "rows": rows,
            },
            "llm_calls": 1 + len(rows),
        },
        params={"model": MODEL, "backend": "vertex", "temperature": 0.0, "tickers": TICKERS,
                "dates_rule": "first trading day of Jan and Jul 2018-2025",
                "horizon": "last trading day <= date+365d",
                "buckets": {"past": "end<=2024-06-30", "recent": "2024-07-01..2025-06-30", "future": "end>=2025-07-01"},
                "legacy_commit": LEGACY_COMMIT, "restructure_commit": RESTRUCTURE_COMMIT,
                "raw_log": str((LOG_DIR / f"contamination_calls_{stamp}.jsonl").relative_to(LAB_ROOT))},
        sources=["https://query1.finance.yahoo.com/v8/finance/chart/ (adjclose)",
                 f"https://github.com/virattt/ai-hedge-fund/tree/{LEGACY_COMMIT}"],
        note="Contamination probe only; persona pipeline (a)-(c) not run because no free data path exists.",
    )


LEGACY = LAB_ROOT / "vendor" / "ai-hedge-fund-legacy"
RUNNER = Path(__file__).resolve().parent / "_legacy_backtest_runner.py"
BT_START, BT_END = "2025-03-03", "2025-10-31"
PERSONAS = ["warren_buffett", "charlie_munger", "ben_graham", "bill_ackman", "cathie_wood", "michael_burry",
            "mohnish_pabrai", "nassim_taleb", "peter_lynch", "phil_fisher", "rakesh_jhunjhunwala",
            "stanley_druckenmiller", "aswath_damodaran"]
SIGN = {"bullish": 1, "neutral": 0, "bearish": -1}


def run_pipeline(out_dir: Path, cap: int = 1300) -> None:
    env = {k: v for k, v in os.environ.items() if k not in ("GEMINI_API_KEY", "GOOGLE_API_KEY")}
    env.setdefault("VERTEX_PROJECT", "gen-lang-client-0060132691")
    env["PYTHONPATH"] = str(LEGACY)
    subprocess.run([str(LEGACY / ".venv" / "bin" / "python"), str(RUNNER), str(out_dir), BT_START, BT_END, str(cap)],
                   cwd=LEGACY, env=env, check=True)


def _yahoo_close(ticker: str) -> pd.Series:
    p1 = int(datetime(2025, 1, 1, tzinfo=timezone.utc).timestamp())
    p2 = int(datetime(2025, 11, 30, tzinfo=timezone.utc).timestamp())
    r = requests.get(YAHOO.format(t=ticker, p1=p1, p2=p2), headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    r.raise_for_status()
    res = r.json()["chart"]["result"][0]
    idx = pd.to_datetime(res["timestamp"], unit="s", utc=True).tz_convert("America/New_York").normalize().tz_localize(None)
    return pd.Series(res["indicators"]["quote"][0]["close"], index=idx).dropna()


def _agreement(cells: pd.DataFrame, agents: list[str]) -> dict:
    """cells: rows=(date,ticker), columns per agent: s*conf/100 and sign."""
    pairs = []
    for i, a in enumerate(agents):
        for b in agents[i + 1:]:
            sa, sb = cells[f"{a}__sig"], cells[f"{b}__sig"]
            va, vb = cells[f"{a}__val"], cells[f"{b}__val"]
            corr = float(va.corr(vb)) if va.std() > 0 and vb.std() > 0 else None
            pairs.append({"a": a, "b": b, "same_signal": float((sa == sb).mean()),
                          "opposite_signal": float(((sa * sb) == -1).mean()), "corr": corr})
    p = pd.DataFrame(pairs)
    return {"n_pairs": len(p), "mean_same_signal": float(p.same_signal.mean()),
            "mean_opposite_signal": float(p.opposite_signal.mean()),
            "mean_corr": float(p["corr"].dropna().mean()), "median_corr": float(p["corr"].dropna().median()),
            "share_pairs_corr_negative": float((p["corr"].dropna() < 0).mean()),
            "most_agreeing": p.sort_values("corr", ascending=False).head(3).to_dict("records"),
            "most_opposed": p.sort_values("corr").head(3).to_dict("records"),
            "pairs": p.to_dict("records")}


def analyze(run_dir: Path) -> None:
    runs = json.loads((run_dir / "rebalances.json").read_text())
    summ = json.loads((run_dir / "summary.json").read_text())
    trades = json.loads((run_dir / "trades.json").read_text())
    pv = pd.DataFrame(json.loads((run_dir / "portfolio_values.json").read_text()))
    pv["Date"] = pd.to_datetime(pv["Date"])
    pv = pv.set_index("Date")

    rows, others = [], []
    for r in runs:
        sigs = r["analyst_signals"]
        all_agents = sorted(k.removesuffix("_agent") for k in sigs if not k.startswith("risk"))
        for t in TICKERS:
            row = {"date": r["date"], "ticker": t, "pm_action": r["decisions"][t]["action"],
                   "pm_qty": r["decisions"][t]["quantity"]}
            for a in all_agents:
                s = sigs[f"{a}_agent"].get(t, {})
                sg = SIGN.get(str(s.get("signal")).lower(), 0)
                conf = float(s.get("confidence") or 0)
                row[f"{a}__sig"], row[f"{a}__val"], row[f"{a}__conf"] = sg, sg * conf / 100, conf
            rows.append(row)
        others = all_agents
    cells = pd.DataFrame(rows)
    nonpersona = [a for a in others if a not in PERSONAS]
    allag = PERSONAS + nonpersona

    def net(agents):
        return cells[[f"{a}__val" for a in agents]].mean(axis=1)

    cells["net_persona"] = net(PERSONAS)
    cells["net_all"] = net(allag)
    cells["n_bull"] = (cells[[f"{a}__sig" for a in PERSONAS]] == 1).sum(axis=1)
    cells["n_bear"] = (cells[[f"{a}__sig" for a in PERSONAS]] == -1).sum(axis=1)
    cells.to_csv(run_dir / "cells.csv", index=False)

    mix = {a: {k: float((cells[f"{a}__sig"] == v).mean()) for k, v in SIGN.items()} | {
        "mean_conf": float(cells[f"{a}__conf"].mean())} for a in allag}
    hold = cells.pm_action.eq("hold")
    near0 = cells.net_persona.abs() < 0.10
    b = {
        "cells": len(cells),
        "pm_action_counts": cells.pm_action.value_counts().to_dict(),
        "pm_hold_share": float(hold.mean()),
        "persona_net_near_zero_share": float(near0.mean()),
        "all_agents_net_near_zero_share": float((cells.net_all.abs() < 0.10).mean()),
        "pm_hold_share_when_near_zero": float(hold[near0].mean()) if near0.any() else None,
        "pm_hold_share_when_not_near_zero": float(hold[~near0].mean()) if (~near0).any() else None,
        "cells_with_ge3_bull_and_ge3_bear": float(((cells.n_bull >= 3) & (cells.n_bear >= 3)).mean()),
        "mean_abs_persona_net": float(cells.net_persona.abs().mean()),
        "mean_persona_split": {"bull": float(cells.n_bull.mean()), "bear": float(cells.n_bear.mean()),
                               "neutral": float(13 - cells.n_bull.mean() - cells.n_bear.mean())},
    }

    # (c) backtest vs equal-weight buy & hold (price-only closes, same as the engine)
    closes = pd.DataFrame({t: _yahoo_close(t) for t in TICKERS + ["SPY"]})
    closes = closes.loc[BT_START:BT_END]
    bh = (closes[TICKERS] / closes[TICKERS].iloc[0]).mean(axis=1)
    eq = pv["Portfolio Value"].astype(float)
    eq = eq[~eq.index.duplicated(keep="last")]
    notional = sum(abs(t["qty"] * t["price"]) for t in trades)
    dd = lambda s: float((s / s.cummax() - 1).min())
    ret = lambda s: float(s.iloc[-1] / s.iloc[0] - 1)
    dret = eq.pct_change().dropna()
    bret = bh.pct_change().dropna()
    c = {
        "period": [BT_START, BT_END], "initial_capital": 100000.0,
        "strategy_total_return": ret(eq), "strategy_max_drawdown": dd(eq),
        "strategy_sharpe_daily_ann": float(dret.mean() / dret.std() * np.sqrt(252)) if dret.std() > 0 else None,
        "buy_hold_eqw_total_return": ret(bh), "buy_hold_eqw_max_drawdown": dd(bh),
        "buy_hold_sharpe_daily_ann": float(bret.mean() / bret.std() * np.sqrt(252)),
        "spy_total_return": ret(closes["SPY"]),
        "n_trades": len(trades), "traded_notional": notional,
        "turnover_traded_over_mean_equity": notional / float(eq.mean()),
        "turnover_traded_over_initial_capital": notional / 100000.0,
        "mean_gross_exposure_frac": float((pv["Gross Exposure"].astype(float) / pv["Portfolio Value"].astype(float)).mean())
        if "Gross Exposure" in pv else None,
        "mean_net_exposure_frac": float((pv["Net Exposure"].astype(float) / pv["Portfolio Value"].astype(float)).mean())
        if "Net Exposure" in pv else None,
        "engine_metrics": summ["engine_metrics"],
        "trades": trades,
        "caveat": "gemini-3.8-flash recalls 12-month returns for windows ending up to mid-2025 almost exactly "
                  "(contamination probe); Mar-Oct 2025 is therefore not a clean out-of-sample period. "
                  "8 monthly decisions only -> no statistical power.",
    }
    # ---- chart data for the video (rebuilt from the saved run + Yahoo closes; no LLM calls) ----
    common = eq.index.intersection(closes.index)
    assert common[0] == eq.index[0] == closes.index[0] and common[-1] == eq.index[-1] == closes.index[-1], \
        (eq.index[[0, -1]], closes.index[[0, -1]])
    spy_n = closes["SPY"] / closes["SPY"].iloc[0]
    ag_n = eq / eq.iloc[0]
    daily = {"dates": [d.strftime("%Y-%m-%d") for d in common],
             "agents": [round(float(ag_n[d] - 1), 5) for d in common],
             "buy_hold_eqw": [round(float(bh[d] - 1), 5) for d in common],
             "spy": [round(float(spy_n[d] - 1), 5) for d in common]}
    for k, ref in (("agents", c["strategy_total_return"]), ("buy_hold_eqw", c["buy_hold_eqw_total_return"]),
                   ("spy", c["spy_total_return"])):
        assert abs(daily[k][-1] - ref) < 1e-5, (k, daily[k][-1], ref)
    if "Net Exposure" in pv:
        ne = (pv["Net Exposure"].astype(float) / pv["Portfolio Value"].astype(float))
        ne = ne[~ne.index.duplicated(keep="last")]
        daily["agents_net_exposure_frac"] = [round(float(ne[d]), 4) for d in common]
    charts = {
        "daily_curves": daily,
        "decision_cells": [{"date": r.date, "ticker": r.ticker, "pm_action": r.pm_action, "pm_qty": int(r.pm_qty),
                            "net_persona": round(float(r.net_persona), 4), "n_bull": int(r.n_bull), "n_bear": int(r.n_bear)}
                           for r in cells.itertuples()],
    }
    values = {
        "data_substituted": True,
        "substitution_doc": "quant_lab/vendor/ai-hedge-fund-legacy/DATA_SUBSTITUTIONS.md",
        "llm_calls": summ["llm_calls"], "llm_errors": summ["llm_errors"],
        "llm_fallbacks_to_default": len(summ["fallbacks"]), "fallbacks": summ["fallbacks"],
        "a_persona_agreement": _agreement(cells, PERSONAS),
        "a_all_agents_agreement": {k: v for k, v in _agreement(cells, allag).items() if k != "pairs"},
        "b_cancellation": b,
        "c_backtest": c,
        "signal_mix": mix,
        "rebalance_dates": [r["date"] for r in runs],
        "charts": charts,
    }
    print(json.dumps({k: values[k] for k in ("llm_calls", "llm_fallbacks_to_default")}, indent=1))
    print(json.dumps({k: v for k, v in values["a_persona_agreement"].items() if k != "pairs"}, indent=1))
    print(json.dumps(b, indent=1))
    print(json.dumps({k: v for k, v in c.items() if k not in ("trades", "engine_metrics")}, indent=1))
    print(json.dumps({a: {k: round(v, 2) for k, v in m.items()} for a, m in mix.items()}))
    write_result(
        "ep05", "persona_vote_experiment", values,
        params={"model": MODEL, "backend": "vertex (ChatVertexAI, location=global, repo default temperature)",
                "tickers": TICKERS, "backtest": [BT_START, BT_END], "cadence": "graph on first trading day of month, hold otherwise",
                "initial_capital": 100000, "margin_requirement": 0.0,
                "near_zero_threshold": 0.10, "signal_value": "sign(bull=+1,neutral=0,bear=-1) x confidence/100",
                "legacy_commit": LEGACY_COMMIT, "patches": ["src/llm/models.py Vertex branch", "src/llm/api_models.json gemini-3.8-flash",
                                                            "src/tools/api.py -> free_data.py"],
                "run_dir": str(run_dir.relative_to(LAB_ROOT))},
        sources=["https://query1.finance.yahoo.com/v8/finance/chart/",
                 "https://data.sec.gov/api/xbrl/companyfacts/", "https://data.sec.gov/submissions/ (Form 4)",
                 f"https://github.com/virattt/ai-hedge-fund/tree/{LEGACY_COMMIT}"],
        note="Agents saw SUBSTITUTE data (Yahoo prices, SEC XBRL fundamentals, SEC Form 4; no news). "
             "Mar-Oct 2025 may be partly memorised by the model (see ai_hedge_fund_probe.contamination).",
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-data", action="store_true")
    ap.add_argument("--contamination", action="store_true")
    ap.add_argument("--pipeline", action="store_true", help="run the legacy graph + engine (~530 LLM calls)")
    ap.add_argument("--analyze", type=Path, help="run dir produced by --pipeline")
    a = ap.parse_args()
    if a.check_data:
        print(check_data())
    if a.contamination:
        contamination()
    if a.pipeline:
        out = LOG_DIR / f"persona_backtest_{BT_START}_{BT_END}"
        run_pipeline(out)
        analyze(out)
    elif a.analyze:
        analyze(a.analyze)


if __name__ == "__main__":
    main()

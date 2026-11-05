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
  STATUS: BLOCKED. financialdatasets.ai returns HTTP 402/401 for every endpoint without a key
          (checked 2026-11-06, incl. AAPL). Per instructions no key/account was created and
          nothing was paid. `--check-data` re-runs that check.

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
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-data", action="store_true")
    ap.add_argument("--contamination", action="store_true")
    a = ap.parse_args()
    if a.check_data:
        print(check_data())
    if a.contamination:
        contamination()


if __name__ == "__main__":
    main()

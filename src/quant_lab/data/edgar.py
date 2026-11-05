"""SEC EDGAR XBRL companyfacts loader (free, no key). Cached under quant_lab/.cache/edgar/."""

from __future__ import annotations

import json
import time

import pandas as pd
import requests

from quant_lab import CACHE_DIR

UA = {"User-Agent": "quant-lab research tool contact@quant-lab.invalid"}
SOURCES: list[str] = []
_last = [0.0]


def _get(url: str, path) -> dict:
    if not path.exists():
        wait = 0.12 - (time.time() - _last[0])          # SEC asks for <= 10 req/s
        if wait > 0:
            time.sleep(wait)
        r = requests.get(url, headers=UA, timeout=60)
        _last[0] = time.time()
        r.raise_for_status()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(r.content)
    if url not in SOURCES:
        SOURCES.append(url)
    return json.loads(path.read_text())


def tickers() -> pd.DataFrame:
    d = _get("https://www.sec.gov/files/company_tickers.json", CACHE_DIR / "edgar" / "company_tickers.json")
    return pd.DataFrame(d.values())


def company_facts(cik: int) -> dict:
    """Empty dict when the filer has no XBRL facts (foreign/new filers return 404)."""
    c = f"{int(cik):010d}"
    try:
        return _get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{c}.json", CACHE_DIR / "edgar" / f"CIK{c}.json")
    except requests.HTTPError as e:
        if e.response is not None and e.response.status_code == 404:
            return {}
        raise


def annual(facts: dict, *concepts: str) -> pd.Series:
    """Fiscal-year values from 10-K filings for the first concept that exists (USD), indexed by FY end date."""
    g = facts.get("facts", {}).get("us-gaap", {})
    for name in concepts:
        if name not in g:
            continue
        rows = [x for x in g[name]["units"].get("USD", []) if x.get("form") == "10-K" and x.get("fp") == "FY"]
        if not rows:
            continue
        df = pd.DataFrame(rows)
        if "start" in df:   # flow concept: keep ~1-year durations only
            dur = (pd.to_datetime(df["end"]) - pd.to_datetime(df["start"])).dt.days
            df = df[dur.between(350, 380)]
        df = df.sort_values("filed").drop_duplicates("end", keep="last")
        return pd.Series(df["val"].to_numpy(dtype=float), index=pd.to_datetime(df["end"])).sort_index()
    return pd.Series(dtype=float)

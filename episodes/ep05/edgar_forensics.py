"""EP05 — what a code-driven research tool can actually do with free SEC data.

Narration claims:
  (a) FinRobot extracts free cash flow and cost of capital from 10-K filings and computes a DCF
      intrinsic value with a margin-of-safety band
  (b) it caught receivables days (DSO) spiking and inventory growing 2x faster than revenue
      "in 10 seconds" and refused the buy

What we do instead of quoting FinRobot:
  forensic screen over the first 100 companies in SEC's company_tickers.json (≈ largest caps),
  latest two fiscal years from 10-K XBRL:
    DSO = AR / revenue x 365 ;  flag_dso = DSO up >= 20% YoY
    flag_inventory = revenue growth > 0 and inventory growth >= 2 x revenue growth
    red_flag = flag_dso AND flag_inventory
  timed end-to-end (cached data excluded from the timing claim)
  illustrative 2-stage DCF for one company with EXPLICIT assumptions — the 10-K contains cash
  flows, NOT a cost of capital; WACC is an analyst assumption, which the script states.
"""

from __future__ import annotations

import time

import numpy as np

from quant_lab.data import edgar
from quant_lab.results import write_result

UNIVERSE_N = 100
REV = ("RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues", "SalesRevenueNet")
AR = ("AccountsReceivableNetCurrent",)
INV = ("InventoryNet",)
CFO = ("NetCashProvidedByUsedInOperatingActivities",)
CAPEX = ("PaymentsToAcquirePropertyPlantAndEquipment",)
DCF_TICKER, WACC, G1, G_TERM, YEARS = "AAPL", 0.09, 0.05, 0.025, 5


def screen_one(cik: int) -> dict | None:
    f = edgar.company_facts(cik)
    rev, ar, inv = edgar.annual(f, *REV), edgar.annual(f, *AR), edgar.annual(f, *INV)
    if len(rev) < 2 or len(ar) < 2:
        return None
    common = rev.index.intersection(ar.index)
    if len(common) < 2:
        return None
    t1, t0 = common[-1], common[-2]
    dso1, dso0 = ar[t1] / rev[t1] * 365, ar[t0] / rev[t0] * 365
    rg = rev[t1] / rev[t0] - 1
    out = {"fy_end": str(t1.date()), "dso": dso1, "dso_prev": dso0, "dso_change": dso1 / dso0 - 1,
           "revenue_growth": rg, "inventory_growth": None}
    if t1 in inv.index and t0 in inv.index and inv[t0] > 0:
        out["inventory_growth"] = inv[t1] / inv[t0] - 1
    out["flag_dso"] = out["dso_change"] >= 0.20
    out["flag_inventory"] = (out["inventory_growth"] is not None and rg > 0 and out["inventory_growth"] >= 2 * rg)
    out["red_flag"] = bool(out["flag_dso"] and out["flag_inventory"])
    return out


def dcf(cik: int) -> dict:
    f = edgar.company_facts(cik)
    cfo, capex = edgar.annual(f, *CFO), edgar.annual(f, *CAPEX)
    idx = cfo.index.intersection(capex.index)
    fcf = (cfo[idx] - capex[idx]).iloc[-3:]
    base = float(fcf.mean())
    flows = [base * (1 + G1) ** t for t in range(1, YEARS + 1)]
    pv = sum(cf / (1 + WACC) ** t for t, cf in enumerate(flows, 1))
    tv = flows[-1] * (1 + G_TERM) / (WACC - G_TERM)
    ev = pv + tv / (1 + WACC) ** YEARS
    sens = {f"wacc_{w:.2f}": (sum(cf / (1 + w) ** t for t, cf in enumerate(flows, 1))
                               + flows[-1] * (1 + G_TERM) / (w - G_TERM) / (1 + w) ** YEARS) / ev - 1
            for w in (0.08, 0.10)}
    return {"ticker": DCF_TICKER, "fcf_last3_usd": {str(k.date()): float(v) for k, v in fcf.items()},
            "base_fcf": base, "enterprise_value_usd": ev, "assumptions": {"wacc": WACC, "growth_5y": G1,
            "terminal_growth": G_TERM, "note": "WACC/growth are assumptions, not 10-K data"},
            "ev_change_if_wacc": sens}


def main():
    tk = edgar.tickers().head(UNIVERSE_N)
    for cik in tk.cik_str:                     # warm cache so timing measures computation only
        edgar.company_facts(cik)
    t0 = time.perf_counter()
    rows = []
    for _, r in tk.iterrows():
        s = screen_one(r.cik_str)
        if s:
            rows.append({"ticker": r.ticker, **s})
    elapsed = time.perf_counter() - t0
    flagged = [r for r in rows if r["red_flag"]]
    print(f"screened {len(rows)}/{UNIVERSE_N} companies in {elapsed:.2f}s; "
          f"DSO flags {sum(r['flag_dso'] for r in rows)}, inventory flags {sum(r['flag_inventory'] for r in rows)}, "
          f"both {len(flagged)}")
    for r in flagged:
        print(f"  {r['ticker']:6} FY{r['fy_end'][:4]} DSO {r['dso_prev']:.0f}->{r['dso']:.0f}d "
              f"rev {r['revenue_growth']:+.1%} inv {r['inventory_growth']:+.1%}")
    cik = int(tk.loc[tk.ticker == DCF_TICKER, "cik_str"].iloc[0])
    d = dcf(cik)
    print(f"DCF {DCF_TICKER}: base FCF ${d['base_fcf']/1e9:.1f}B -> EV ${d['enterprise_value_usd']/1e12:.2f}T "
          f"(WACC 8%: {d['ev_change_if_wacc']['wacc_0.08']:+.0%}, 10%: {d['ev_change_if_wacc']['wacc_0.10']:+.0%})")
    write_result("ep05", "edgar_forensics", {"screened": len(rows), "seconds_compute_only": elapsed,
                 "n_flag_dso": int(sum(r["flag_dso"] for r in rows)),
                 "n_flag_inventory": int(sum(r["flag_inventory"] for r in rows)),
                 "red_flags": flagged, "all": rows, "dcf_example": d},
                 params={"universe": f"first {UNIVERSE_N} of SEC company_tickers.json", "dso_up": 0.20,
                         "inventory_vs_revenue": 2.0}, sources=list(edgar.SOURCES[:3]) + ["... companyfacts per CIK"])


if __name__ == "__main__":
    main()

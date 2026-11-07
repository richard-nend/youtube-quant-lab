"""EP06 — absorption bot: taker vs maker vs "absorption -> failure" with the 5x cost rule.

Narration claims:
  (a) absorption bot, market orders: ~700 trades, win 64%, result -42%, round-trip cost 0.13%
      ("beat the market, lost to the exchange")
  (b) limit-only entries suffer adverse selection; switching the backtest from "touch" to
      "through" fills exposes it
  (c) final rules (5x cost filter, wait for failure after absorption, stop beyond the sweep wick)
      leave "a gentle but real" upward curve

Pre-registered definitions (5m BTCUSDT perp bars, 2025-11-01 -> 2026-10-31):
  delta_t       = taker_buy_volume - taker_sell_volume (exact, from Binance kline fields)
  absorption    = |z(delta)| >= 2.5 over a rolling 1-day window AND |close-open| <= 0.25 x ATR14
  direction     = fade the aggressor (buy aggression absorbed -> short)
  V1 taker      : market at next open, TP 0.75 ATR, SL 1.5 ATR, max hold 24 bars (2h)
  V2 maker/touch: limit at the absorption bar close, same TP/SL, filled on touch
  V3 maker/thru : same as V2 but filled only if price trades THROUGH the limit
  V4 failure    : after absorption, wait up to 6 bars for a close beyond the absorption bar's
                  far extreme (failure of the aggressor); market entry next open; stop beyond the
                  absorption bar's wick (+0.1 ATR); target 2R; skip unless target distance >=
                  5 x round-trip cost
Position sizing for the equity curve: 1x equity notional per trade (no leverage).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_lab.data import binance
from quant_lab.engine.backtest import Signal, simulate
from quant_lab.engine.costs import CostModel
from quant_lab.metrics import equity_curve, summarize
from quant_lab.results import ohlc_rows, write_result

START, END = "2025-11-01", "2026-10-31"
Z_MIN, BODY_ATR, ZWIN = 2.5, 0.25, 288
TP_ATR, SL_ATR, HOLD = 0.75, 1.5, 24
FAIL_WAIT, WICK_BUF, V4_RR, COST_MULT = 6, 0.1, 2.0, 5.0
COSTS = CostModel()


def atr(b, n=14):
    tr = pd.concat([b.high - b.low, (b.high - b.close.shift()).abs(), (b.low - b.close.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def signals(b: pd.DataFrame):
    """(absorb idx, V1 taker signals, V2 limit signals, V4 failure signals, skipped_by_cost_rule)."""
    o, h, l, c = (b[x].to_numpy() for x in ("open", "high", "low", "close"))
    a = atr(b).to_numpy()
    delta = (2 * b.taker_buy_volume - b.volume)
    z = ((delta - delta.rolling(ZWIN).mean()) / delta.rolling(ZWIN).std()).to_numpy()
    body = np.abs(c - o)
    absorb = np.flatnonzero((np.abs(z) >= Z_MIN) & (body <= BODY_ATR * a))
    side_of = lambda i: -1 if z[i] > 0 else 1        # fade the aggressor

    v1, v2 = [], []
    for i in absorb:
        s = side_of(i)
        entry_ref = c[i]
        v1.append(Signal(bar=i, side=s, stop=entry_ref - s * SL_ATR * a[i], target=entry_ref + s * TP_ATR * a[i]))
        v2.append(Signal(bar=i, side=s, stop=entry_ref - s * SL_ATR * a[i], target=entry_ref + s * TP_ATR * a[i],
                         entry="limit", limit_price=entry_ref))

    rt = COSTS.round_trip("taker", "taker")
    v4, skipped_cost = [], 0
    for i in absorb:
        s = side_of(i)
        for j in range(i + 1, min(i + 1 + FAIL_WAIT, len(b) - 1)):
            failed = (c[j] < l[i]) if s < 0 else (c[j] > h[i])
            if failed:
                entry_est = o[j + 1]
                stop = (h[i] + WICK_BUF * a[i]) if s < 0 else (l[i] - WICK_BUF * a[i])
                risk = (stop - entry_est) * -s
                if risk <= 0:
                    break
                target = entry_est + s * V4_RR * risk
                if abs(target - entry_est) / entry_est < COST_MULT * rt:
                    skipped_cost += 1
                    break
                v4.append(Signal(bar=j, side=s, stop=stop, target=target))
                break
    return absorb, v1, v2, v4, skipped_cost


N_EXAMPLES, PRE, POST = 3, 12, 24


def absorption_charts(b: pd.DataFrame, absorb, v1_trades: pd.DataFrame) -> dict:
    """Event timestamps + 5m candle windows around the FIRST N_EXAMPLES events of the year
    (chronological, not chosen by outcome). Video only; no effect on the numbers above."""
    o, c = b.open.to_numpy(), b.close.to_numpy()
    a = atr(b).to_numpy()
    delta = (2 * b.taker_buy_volume - b.volume)
    z = ((delta - delta.rolling(ZWIN).mean()) / delta.rolling(ZWIN).std()).to_numpy()
    events = {"t": [b.index[i].isoformat() for i in absorb],
              "z": [round(float(z[i]), 2) for i in absorb],
              "body_over_atr": [round(float(abs(c[i] - o[i]) / a[i]), 3) for i in absorb]}
    v1_by_sig = v1_trades.set_index("signal_time") if not v1_trades.empty else None
    ex = []
    for i in absorb[:N_EXAMPLES]:
        s0 = max(i - PRE, 0)
        tr = None
        if v1_by_sig is not None and b.index[i] in v1_by_sig.index:
            r = v1_by_sig.loc[b.index[i]]
            tr = {"fill_i": int(b.index.get_loc(r.entry_time) - s0), "exit_i": int(b.index.get_loc(r.exit_time) - s0),
                  "entry_px": round(float(r.entry_px), 2), "exit_px": round(float(r.exit_px), 2),
                  "exit_reason": r.exit_reason, "net_ret": round(float(r.net_ret), 6)}
        ex.append({"event_i": int(i - s0), "z": round(float(z[i]), 2), "fade_side": -1 if z[i] > 0 else 1,
                   "atr": round(float(a[i]), 2), "v1_trade": tr,
                   "bars": ohlc_rows(b, s0, i + POST + 1, extra={"delta": delta.to_numpy(), "z": np.nan_to_num(z)})})
    return {"absorption_events": events,
            "examples_how_chosen": f"first {N_EXAMPLES} absorption events of the year (chronological, not chosen by outcome)",
            "examples": ex}


def main():
    b = binance.klines("BTCUSDT", "5m", START, END, market="um")
    absorb, v1, v2, v4, skipped_cost = signals(b)
    rt = COSTS.round_trip("taker", "taker")

    runs = {
        "V1_taker": simulate(b, v1, COSTS, max_hold=HOLD),
        "V2_maker_touch": simulate(b, v2, COSTS, fill="touch", limit_ttl=3, max_hold=HOLD),
        "V3_maker_through": simulate(b, v2, COSTS, fill="through", limit_ttl=3, max_hold=HOLD),
        "V4_failure_5x_rule": simulate(b, v4, COSTS, max_hold=HOLD * 2),
    }
    results = {"absorption_events": int(len(absorb)), "v4_skipped_by_cost_rule": skipped_cost,
               "round_trip_cost_taker": rt}
    print(f"absorption events {len(absorb)}, round-trip taker cost {rt:.4%}, V4 skipped by 5x rule {skipped_cost}")
    print(f"{'variant':20} {'trades':>6} {'win':>6} {'return':>8} {'preCost':>8} {'costPaid':>9} {'maxDD':>7}")
    for k, t in runs.items():
        s = summarize(t)
        if s.get("trades", 0):
            eq = equity_curve(t["net_ret"]).to_numpy()
            s["equity_curve_sampled"] = eq[:: max(len(eq) // 200, 1)].round(4).tolist()
            s["gross_equity_curve_sampled"] = equity_curve(t["gross_ret"]).to_numpy()[:: max(len(eq) // 200, 1)].round(4).tolist()
            print(f"{k:20} {s['trades']:>6} {s['win_rate']:>6.1%} {s['total_return']:>8.1%} "
                  f"{s['total_return_before_costs']:>8.1%} {s['total_cost_paid']:>9.1%} {s['max_drawdown']:>7.1%}")
        results[k] = s
    # adverse selection: among V2 touch-fills, how did trades that V3 would NOT have filled do?
    t2, t3 = runs["V2_maker_touch"], runs["V3_maker_through"]
    if not t2.empty:
        only_touch = t2[~t2["signal_time"].isin(t3["signal_time"])] if not t3.empty else t2
        results["adverse_selection"] = {
            "touch_only_fills": int(len(only_touch)),
            "touch_only_win_rate": float((only_touch.net_ret > 0).mean()) if len(only_touch) else None,
            "through_fills_win_rate": float((t3.net_ret > 0).mean()) if len(t3) else None,
        }
        print("adverse selection:", results["adverse_selection"])

    results["charts"] = absorption_charts(b, absorb, runs["V1_taker"])
    assert len(results["charts"]["absorption_events"]["t"]) == results["absorption_events"]
    write_result("ep06", "absorption_bot_backtest", results,
                 params={"symbol": "BTCUSDT", "interval": "5m", "range": [START, END], "z_min": Z_MIN,
                         "body_atr": BODY_ATR, "tp_atr": TP_ATR, "sl_atr": SL_ATR, "hold_bars": HOLD,
                         "fail_wait": FAIL_WAIT, "v4_rr": V4_RR, "cost_mult": COST_MULT,
                         "costs": COSTS.as_dict(), "position": "1x equity notional"},
                 note="bar-based engine (5m), not a tick engine; 'through' fill approximates queue position")


if __name__ == "__main__":
    main()

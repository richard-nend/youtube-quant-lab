# Claims ledger: every number in EP01–EP06 narration

Status: ✅ verified by code · ❌ contradicted by code/data · 🔶 partially true · ⏳ not yet run · 📎 needs external source (see `sources.md`)
Results JSON: `src/data/results/epXX.json` (each entry has script, params, seed, data URLs).

## EP04

| Claim (narration) | Status | Measured | Script |
|---|---|---|---|
| 1,000 traders, 60% win, 1.5x/2.0x martingale → 994 ruined | ❌ | 2.0x: **32.6%** (326/1000) · 1.5x: **1.8%** | `ep04/martingale_montecarlo.py` |
| (what WOULD give ~99%) | — | 50% win, 2.0x, 2,000 trades → 96.6% · 10,000 trades → 100% | same, `grid` |
| 7 losses → next bet 12,800, cumulative 25,500 | 🔶 | next bet 12,800 ✔ · lost so far **12,700** · 25,500 includes the 8th bet | `ep04/martingale_table.py` |
| 7 losses → 128x | ✅ | 2^7 = 128 | same |
| 7 opposite moves in a row < 1% | 🔶 | 0.78% at 50/50 ✔ · 0.16% at 60% win (state the assumption) | same |
| Shorts: 99 wins = $500, one crash wipes $50,000 | ❌ | implies $5.05 base → needs **13** straight losses to wipe $50k, not 7 | same |
| LUNA: 12 consecutive red 15m candles | ✅ | **13** in a row, 2022-05-05 12:45–15:45 UTC, −6.6% (FOMC day, before the UST depeg) | `ep04/blackswan_scanner.py` |
| FTX: −24% in 2 hours | ❌ | worst 2h **−14.1%** (2022-11-08 17:34 UTC) · worst 24h −18.6% · week high→low −27.0% | same |
| 300-day $40k bot, $120k wiped in 3h | 📎 | no source found in production logs | — |
| Kelly/anti-martingale → ruin 0%, "the only answer" | ❌ overclaim | p=0.55, 1,000 bets, 10k paths: full Kelly ruin(<1%) **0.68%**, P(−50% drawdown) **100%**; half Kelly ruin 0 but P(−50% DD) 85.7%; **2x Kelly (edge overestimated) ruin 47.4%**; Kelly with no real edge ruin 66.4%. Martingale ruin 71.1%. Fixed 2%: ruin 0, P(−50% DD) 1.8% | `ep04/sizing_comparison.py` |
| Grid: steady in ranges, stuck long in trends | ✅ | 30-day windows 2024-01→2026-10: range (11) mean **+3.2%**; downtrend (5, BTC −18%) mean **−10.7%**, worst −22%, all 10 levels stuck | `ep04/grid_bot_sim.py` |
| Fix: ATR spacing | ✅ partial | downtrend −10.7% → **−3.9%**, but range profit shrinks +3.2% → +0.8% (trade-off) | same |
| Fix: 3-consecutive-fill lock | ✅ partial | downtrend −10.7% → **−7.7%**; range +3.2% → +2.6% | same |

## EP06

| Claim | Status | Measured | Script |
|---|---|---|---|
| Dec 2024 $100k break: OI **decreased** during the green candle (short covering) | ❌ | Binance BTCUSDT perp: OI **rose** in every window (+6.5% / +3.3% / +2.2% vs price +6.7% / +5.2% / +4.3%) | `ep06/dec2024_oi_autopsy.py` |
| Hours later, liquidation cascade | 🔶 | flash low **$90,200** at 2024-12-05 22:28 UTC, ~20h after the cross, −13.8% from $104,630 peak. Binance OI by window: last 30 min into the low **+0.5%** (JSON `post_cross_drop.oi_change_pct`); 03:10→23:00 **−1.2%**; 03:10→next day 01:00 **−4.3%** (95,248 → 91,126 BTC). Single-exchange view; no big OI flush at the wick itself | same |
| isBuyerMaker sign trap | ✅ (unit-tested) | `orderflow.signed_volume` + `tests/test_core.py` | |
| Absorption bot, market orders: 700 trades, 64% win, −42%, 0.13% round trip | 🔶 | 5m BTC perp, 1 year: **211 trades, win 55.5%, −19.6%**; before costs −0.7% (≈ flat) → **costs alone = −21.1%**. Mechanism ("didn't lose to the market, lost to fees") holds; the 700/64%/−42% figures do not | `ep06/absorption_bot_backtest.py` |
| Limit-only → adverse selection ("fills only when I'm wrong") | ✅ (ticks) | aggTrades, 6 days (1st of Jan–Jun 2026), passive limits $1 from last price every 60 s, 60 s TTL: **filled orders −1.24 bp after 5 min (45% profitable, before the 2 bp maker fee); unfilled orders would have made +4.20 bp (66% profitable)**. Touch vs through barely differs at this offset | `ep06/adverse_selection_ticks.py` |
| (bar-level version) | — | at 5m bars the effect is invisible (only 4 touch-only fills) — resolution matters | `ep06/absorption_bot_backtest.py` |
| Final rules (5x cost filter, wait for failure, stop beyond wick) → gentle real upward curve | ❌ | **80 trades, win 37.5%, −8.5%** (before costs +0.7%); 64 setups skipped by the 5x rule | same |

## EP02 (external sources: `sources.md`)

| Claim | Status | Fact |
|---|---|---|
| "Real traders" ran Jev 24h: 731 trades, −$3,150, fees $1,650 | 🔶 | Numbers match u/Obside_AI, r/ai_trading, 2026-09-19 — but **$100k PAPER account, CME micro futures (MNQ/MBT/MGC)**, one trader, author says "one day is far too short". Also: ~$1,500 directional loss, only 21% trades profitable after fees |
| Jev/Laya decide in 0.1 s | 🔶 | Jev vendor claim 70–500 ms; **that bot measured 779 ms avg**. Laya (Apache-2.0, 421M) card claims ~33–40 ms/question on T4 |
| Early +$400, 18h chop, 0.1% per flip | ❌ | not in the post — invented |
| Hybrid: 731 → 8 trades, 4 traps blocked | ❌ | invented — must be an actual experiment or removed |
| TradingAgents bear prompt forces "≥3 contradictions" | ❌ | no version of `bear_researcher.py` contains it |
| GPT 5.2 s latency, −42% in 18 min, 82.4% | ⏳ | unsourced; needs `llm_latency_bench.py` / replay |

## EP05

| Claim | Status | Fact |
|---|---|---|
| ai-hedge-fund 25,000 stars | ❌ | 64,502 (2026-11-05) |
| 15 investor personas | ❌ | 13 personas + 4 analysis agents (last README before Aug-2026 restructure) |
| backtest 140% / live −18% / turnover 350% / MDD 11% | ❌ | no source anywhere — invented unless we run it |
| "the LLM already knows which stocks rose" (pretraining contamination) | ✅ (strong) | gemini-3.8-flash, no data in prompt, 5 tickers × 16 dates: 12-month returns for windows ending ≤ 2024-06 recalled with **MAE 0.93pp, sign 100%** (NVDA 2022-07 actual +191.5% → model +191.5%); windows ending ≥ 2025-07: **MAE 25.3pp, sign 60%** (worse than always guessing "up"). `ep05/ai_hedge_fund_probe.py` |
| "Persona votes cancel out → ambiguous HOLD" (consensus paradox) | ❌ as stated | legacy 6c41ae8, 13 personas, **substitute free data** (Yahoo + point-in-time SEC XBRL/Form 4, no news), AAPL/GOOGL/MSFT/NVDA/TSLA, 8 monthly decisions Mar–Oct 2025, 529 Vertex calls: PM chose HOLD in **65%** of cells, but near-zero cancellation only **5%**; personas leaned **bearish** (avg 5.7 bear / 2.0 bull / 5.3 neutral; Graham 100% bearish). Pairwise agreement 42%, mean signal corr 0.40 |
| ai-hedge-fund backtest (claimed 140% / −18% / turnover 350%) | ❌ | agents **−1.3%** (MDD −9.6%, Sharpe −0.09) vs equal-weight buy & hold **+50.6%**, SPY +16.8%; avg net exposure **−35%** (short AAPL/MSFT/TSLA in a strong up-market); 14 trades, turnover 1.15× equity. 8 decisions = no statistical power; Mar–Oct 2025 may be partly memorized |
| FinRobot 10-K RAG + code DCF | ✅ | DCF is fixed Python code, not LLM-written |
| "extracts cost of capital from the 10-K" | ❌ | 10-K has cash flows, not WACC — WACC is an analyst assumption. Our AAPL DCF (3y avg FCF $114.5B): EV $2.01T at 9%; **±1pp WACC moves EV +18% / −13%** (`ep05/edgar_forensics.py`) |
| Caught DSO spike + inventory 2x revenue "in 10 seconds" | 🔶 (own screen) | free SEC XBRL screen of top-100 filers: 55 screenable, **3.6 s** compute; DSO flags 9, inventory flags 15, **both: 1 (DELL FY2026: DSO 39→57d, revenue +18.8%, inventory +55.4%)** — a red flag is a question, not proof of fraud (AI-server build-out) |

## EP03 (`ep03/smc_backtest.py`, BTCUSDT perp 15m, 2025-11-01 → 2026-10-31, 3R target, fill="through", VIP0 costs)

| Claim | Status | Measured |
|---|---|---|
| Mechanical FVG+OB entry → win 34.2% | 🔶 | **30.3%** (1,954 trades, causal) · library-naive 30.8%. Gross edge slightly positive (+14.8% before costs at 10% notional) but **costs flip it negative** (−0.31R/trade): FVG stops are tiny, so fees are a large fraction of R |
| Filters → trades −70%, win 52%, R:R 1:3.2, expectancy turns positive | ❌ | cumulative filters cut trades **1,935 → 10 (−99.5%)**; win rate stays **30–32%** at every step; expectancy improves only to ≈ −0.01R; 10 trades = no statistical power |
| −25% one-month loss before filters | ⏳ | monthly table in results JSON |
| Shorts R:R 1:4 vs long-form 1:3 | ❌ | inconsistent; tested with 3R |
| `smartmoneyconcepts` is look-ahead free | ❌ (new finding) | `fvg()` reads candle i+1, `swing_highs_lows()` reads swing_length/2 future candles, `ob()` **erases OBs that later fail** (survivorship). Naive use must be avoided — good EP01-style segment |

## EP01 (`ep01/fake_backtest_tricks.py`, BTCUSDT perp 1h, 2025-11-01 → 2026-10-31, buy & hold −29.2%)

| Claim | Status | Measured |
|---|---|---|
| Heikin-Ashi fills inflate backtests to "thousands of %" | ✅ | same HA colour-flip strategy: **HA-price fills +25,178%** vs real next-open fills **−20.1%** (−95.7% with fees), 2,243 trades |
| Repainting arrows look like a "god indicator" (95% win) | ✅ | pivot arrows on history: **96.9% win** (859 trades) vs acting when confirmed: **43.1%** (−34.1%), with fees **35.3% / −78.4%** |
| Fees turn it into a loss | ✅ | see above |

## Answer experiments (round 2, pre-registered: `quant_lab/PREREGISTRATION.md`, commit 197476e)

Verdict = PASS (net expectancy/trade > 0 on BTC-H, BTC-D, ETH-H, ≥100 trades each) · WEAK (BTC-H only) · FAIL.
Expectancy = mean net return per trade at fixed notional (Amendment 1); R-expectancy also in JSON.

| Hypothesis | Verdict | Measured | Script |
|---|---|---|---|
| EP03: skip FVG unless stop ≥ 10× round-trip cost (0.85%) | ❌ FAIL (all TFs) | 15m: BTC-D +0.169%/trade (110) but BTC-H −0.023% (99) · 1h: BTC-H −0.089% · 4h: <100 trades | `ep03/cost_aware_smc.py` |
| (same run) FVG 1h, no rule | PASS on paper | BTC-D +0.201% (CI +0.05…+0.36) · BTC-H **+0.008%** (CI −0.13…+0.15) · ETH-H +0.106% · R-expectancy on BTC-H −0.09R → not distinguishable from 0 | same |
| EP06 (a): aggression filter improves 1h Donchian breakout | ✅ | BTC-H −0.193% → **−0.127%** per trade | `ep06/flow_confirmed_breakout.py` |
| EP06 (b): filtered breakout has an edge | ❌ FAIL | BTC-H −0.127% (CI −0.37…+0.13), BTC-D +0.185%, ETH-H +0.006% | same |
| EP01: report card (6 checks) on 10 series strategies | — | every strategy ≥1 flag; most common: cost/risk (13–44% of 1R); delay test collapses both EP01 tricks and both 1h breakouts; holdout sign flips for EP03 5-filter, EP03 cost rule, both EP06 breakouts | `protocol/report_card.py` |
| EP02 (a) veto worst-case > rule worst-case | ✅ | 23 events, 20x: worst event **−17.3%** (veto, 5 runs/event) vs **−100%** (RSI rule, 1 liquidation) | `ep02/llm_crash_replay.py --answer` |
| EP02 (b) veto mean ≥ rule mean | ✅ | −0.74% vs −3.22% per event (both negative: no arm makes crash-buying profitable) | same |
| EP02 (c) veto worst-case > rule + fixed 1.5% stop | ✅ | −17.3% vs −39.8% (fixed stop mean −3.71%: stop alone did not help the mean) | same |
| (exploratory, NOT pre-registered) random veto at the LLM's rate 75% (430/573) | — | 460 runs: mean −1.30%, worst event −50.1%, worst run −74.3%, 5th pct −18.4% vs LLM veto −11.6% → ~3/4 of the mean gain is just trading less; the tail protection is not. Confound: approved LLM entries also carry an LLM-chosen stop. LLM veto median latency 10.4 s | same, arm `rule_randveto` |

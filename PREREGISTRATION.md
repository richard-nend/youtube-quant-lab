# Pre-registration: "answer" experiments (written 2026-11-06, before any of these were run)

The first round of experiments showed what does NOT work. This round tests the fixes we
want to recommend on screen. Every rule, parameter, data period and pass/fail criterion
below is fixed before the code is run. Results are reported whatever they are; a FAIL is
reported as a FAIL and the episode's recommendation changes accordingly.

The commit that adds this file is the timestamp. Any later change to this file is
listed in the "Amendments" section at the bottom with the reason.

## Common rules

- **Instruments / periods**
  - `BTC-D` (discovery): BTCUSDT perp, 2025-11-01 → 2026-10-31. Already seen in round 1.
  - `BTC-H` (holdout, primary): BTCUSDT perp, 2024-11-01 → 2025-10-31. Not used to design any EP03/EP06 rule.
  - `ETH-H` (holdout, secondary): ETHUSDT perp, 2024-11-01 → 2026-10-31.
- **Engine**: `quant_lab.engine.backtest.simulate`, unchanged. Market entries fill at the next bar's open.
  Limit entries fill only when price trades through. When a bar touches both stop and target, the stop counts first.
- **Costs**: Binance VIP0. Taker 0.05% + 1.5 bp slippage; maker 0.02%.
- **Verdict per hypothesis**
  - `PASS`: net expectancy > 0 on BTC-H, BTC-D **and** ETH-H, each with ≥ 100 trades.
  - `WEAK`: net expectancy > 0 on BTC-H with ≥ 100 trades, but not on all three.
  - `FAIL`: otherwise.
  - Also reported: bootstrap 95% CI of the mean net return per trade (10,000 resamples, seed 20261106).

## H-EP03: cost-aware SMC (the cost-to-risk rule)

Round 1 finding: FVG stops are so tight that fees ate ~0.3R per trade.

- **Signals**: round-1 `B_causal_fvg_only`, unchanged.
  - Causal FVG.
  - Limit order at the near edge of the gap.
  - Stop at the far extreme of the displacement candle.
  - Target 3R.
  - Limit valid 20 bars; max hold 96 bars.
- **Rule**: skip any signal whose stop distance is smaller than `10 × round_trip(maker entry, taker stop)`.
  This caps the cost at ≤ 10% of 1R.
- **Timeframes**: 15m, 1h, 4h. Each timeframe is reported with and without the rule.
- **Hypothesis**: with the rule, net expectancy > 0.
  - One timeframe meeting `PASS` makes the episode's answer "SMC only above that timeframe, with the rule".
  - If no timeframe passes, the on-screen answer is "the cost-to-risk check rejects this strategy".

## H-EP06: order flow as a confirmation filter, not a fade signal

Round 1 finding: fading 5m absorption is about flat before costs and deeply negative after.

- **Bars**: 1h. Delta = `2 × taker_buy_volume − volume`.
  Delta z-score over a rolling 720 bars (30 days), using only past bars.
- **T0 (baseline, no order flow)**: Donchian breakout.
  - Entry: close above the highest high of the previous 20 bars → long at the next open.
    The mirror rule gives shorts.
  - Stop 2 × ATR14; target 4 × ATR14; max hold 48 bars.
- **T1 (aggression confirms)**: T0, keeping only breakouts whose breakout-bar delta z ≥ +1 in the breakout direction.
- **T2 (aggression opposes)**: T0, keeping only breakouts whose delta z ≤ −1 against the direction.
  This is a control; it is expected to be worst.
- **Hypotheses**:
  - (a) T1 net expectancy > T0 net expectancy on BTC-H.
  - (b) T1 is `PASS`.
  - If (a) holds but (b) fails, the answer is "order flow improves a trend system but does not create an edge by itself".

## H-EP02: LLM as a brake, not a trader

Round 1 pilot: a fixed RSI rule did better than the LLM trader. The claim we want to make is
"rules decide, the LLM only vetoes". It needs a control without any LLM.

- Same 23 events, window, account (20x), costs and fills as `llm_crash_replay.py`.
- **New arms**
  - `rule_stop` (no LLM): the `rsi_rule` arm plus a fixed 1.5% stop from the average entry. Deterministic, 1 run per event.
  - `rule_veto` (LLM): the `rsi_rule` arm. Each time the rule wants to `open_long` or `add`, the LLM sees the
    same market prompt plus the proposed action and returns `approve` (true/false) and a `stop_pct`.
    - Veto → hold.
    - Approve → the action is taken with the LLM's stop.
    - No LLM call on other steps.
    - Same model `gemini-3.8-flash`, 5 runs per event.
- **Hypotheses** (over the 23 events; worst case = the minimum over events of the per-event mean):
  - (a) `rule_veto` worst-case return > `rsi_rule` worst-case return.
  - (b) `rule_veto` mean return ≥ `rsi_rule` mean return.
  - (c) `rule_veto` worst-case > `rule_stop` worst-case.
  - If (c) fails, the on-screen answer is "a fixed stop does the LLM's job for free".

## H-EP01: the backtest audit (report card)

`quant_lab.protocol.audit` runs the same checks on every strategy in the series.
The checks are fixed here; the thresholds are rules of thumb, stated as such on screen.

1. **Delay test**: shift every signal one bar later. If net return falls by more than half, flag look-ahead or a fragile edge.
2. **Cost-to-risk**: flag if the median round-trip cost exceeds 10% of 1R (risk).
3. **Cost stress**: net expectancy at 2× costs.
4. **Sample size**: flag if there are fewer than 100 trades or the bootstrap 95% CI of mean net return includes 0.
5. **Out-of-sample**: same rules on the holdout period. Flag if the sign of net expectancy flips.
6. **Concentration**: flag if the 5 best trades make more than 50% of the total net profit.

Report card rows: the EP01 HA fills, EP01 repainting pivots, EP03 B/C5/cost-aware, EP06 V1/V4/T0/T1.

## H-EP07: overfitting, the zero-fee illusion, and walk-forward (written 2026-11-07, before any EP07 code was run)

EP07 asks how much of a "best backtest" is the search itself. Its own periods (not the BTC-D/BTC-H sets above):
`IS` = BTCUSDT perp 1h, 2024-11-01 → 2025-10-31 (optimise here); `OOS` = 2025-11-01 → 2026-10-31 (the year after).

- **Strategy family (fixed grid)**: 1h moving-average cross, long and short.
  - Signal on the bar where sign(SMA_fast − SMA_slow) changes; market entry at the next open.
  - Stop k × ATR14, target r × k × ATR14, max hold 48 bars. One position at a time (engine default).
  - fast ∈ {5, 10, 15, 20, 30, 40, 50}, slow ∈ {20, 30, 50, 80, 120, 160, 200} with fast < slow,
    k ∈ {1, 1.5, 2, 3}, r ∈ {1, 2, 3}. Every combination is run; the count is reported.
  - Selection score: sum of per-trade net returns (fixed notional), combos with < 30 trades excluded.
- **E1, the illusion**: pick the best combo on IS with zero costs. Report it on IS with zero costs,
  on IS with real costs, and on OOS with real costs. Also the share of all combos with positive sum,
  zero vs real costs.
- **E2, selection bias (permutation test)**: 200 synthetic IS years. Each one shuffles whole IS bars
  (open/high/low/close relative to the previous close, with taker volume) and rebuilds prices from the first close.
  Seed 20261107. The same grid runs on each; the best zero-cost score per synthetic year gives a noise distribution.
  p = share of synthetic bests ≥ the real IS best.
- **E3, the answer: walk-forward**.
  - Months 2023-11 … 2026-10 (36 months), BTCUSDT and ETHUSDT perp 1h, data from 2022-11-01.
  - At the start of each month, pick the best combo by real-cost score over the previous 12 months.
    It must have ≥ 30 trades there; if none qualifies, sit out.
  - Trade that month with it. Only signals inside the month are taken; an open trade may exit after month end.
  - Report the concatenated out-of-sample trades and the mean in-sample score of the chosen combos per trade,
    compared with the realised one ("decay").
- **Hypotheses (predictions, reported whether right or wrong)**:
  - (a) E1: the zero-cost IS best has net expectancy ≤ 0 on OOS with real costs.
  - (b) E2: p > 0.05, i.e. the real IS best is not unusual against the best of the same grid on shuffled prices.
  - (c) E3 verdict: `PASS` if the walk-forward net expectancy > 0 on BTC and ETH, each with ≥ 100 trades;
    `WEAK` if BTC only; else `FAIL`. Bootstrap CI as above.
- **Engine note**: market entries only, so the same-bar limit-fill issue does not apply.
  Trades where the entry open is already beyond the stop are counted and reported.

## H-EP07B: assembling the survivors of EP01–EP06, walk-forward + live paper test (written 2026-11-07, before any of this code was run)

EP03 and EP06 told viewers their best candidates would be re-tested in EP07. This is that test.
Engine fix first (commit before this one): limit fills no longer count a same-bar target; an open beyond the stop exits at that open.

- **Data**: BTCUSDT and ETHUSDT perp, from 2022-11-01. Out-of-sample months 2023-11 … 2026-10 (36). Costs as in Common rules.
- **Entry candidates (existing code, parameters unchanged unless stated)**:
  - `FVG-1h`: EP03 `B_causal_fvg_only` on 1h, no stop-width rule (the EP03 paper PASS).
  - `FVG-4h`: the same on 4h (EP03 WEAK).
  - `FLOW-1h`: EP06 `T1` (Donchian 20 breakout + breakout-bar delta z ≥ +1, stop 2 ATR, target 4 ATR, hold 48 bars, z over 720 bars).
  - `FLOW-4h`: `T1` on 4h with z over 180 bars (30 days) and hold 12 bars (48 h); everything else the same.
- **Layers, added one at a time and each reported**:
  - `L0`: each candidate alone, fixed notional, over all 36 months (no selection).
  - `L1 selection`: at the start of each month, trade only the candidate with the best net sum over the previous 12 months
    (≥ 30 trades there; otherwise sit out). This is the walk-forward.
  - `L2 brake (Laya)`: `convaiinnovations/laya-multilingual` rev 1720e3e3, local CPU, `laya` 0.4.22.
    - For every L1 entry, ask one yes/no ("noul") question.
      - The state holds only past-bar facts: strategy type, side, last 24 h and 7 d return, ATR %, breakout-bar delta z,
        and distance from the 20-day high/low.
      - It holds no ticker, no dates and no prices, so the model cannot recall the outcome (EP05 lesson).
    - Question: "Should this trade entry be blocked because conditions are dangerous for it?" Veto if noul ≥ 0.5.
    - Control: random veto of L1 entries at Laya's observed veto rate, 20 runs, seed 20261107.
    - Robustness: the same with a second phrasing, "Is this a bad moment to open this position?", reported, not used for verdicts.
  - `L3 sizing (EP04)`: risk 2% of current equity per trade (notional = 2% / stop distance), leverage capped at 3×, compounding.
- **Reported for every layer**: trades, win rate, net expectancy (fixed notional) with bootstrap CI, the report card (6 checks),
  the worst month, and the max drawdown. For L3 also final equity, plus the probability of a −50% drawdown
  (block bootstrap of monthly trade sequences, 10,000 resamples).
- **Hypotheses (predictions)**:
  - (a) `L1` verdict (PASS/WEAK/FAIL as in Common rules; sets = BTC and ETH out-of-sample, primary BTC).
  - (b) Laya brake: `L2` max drawdown and worst month are smaller than `L1`'s, and smaller than the random-veto median.
  - (c) Laya brake: `L2` net expectancy > random-veto median expectancy (does the model pick, or does it only trade less?).
  - (d) `L3`: probability of a −50% drawdown < 5%.
- **Live paper test (H-EP07-LIVE)**:
  - Start: the first full UTC hour after the live runner is committed. Duration: 13 weeks.
  - Runs the exact `L1 + L2 + L3` procedure, unchanged, on live Binance data: hourly bars, simulated fills by the same engine rules,
    and paper equity of 10,000 USDT. No real money.
  - Every signal, Laya answer, fill and equity value is appended to `live/` and committed daily to the public repo.
  - Nothing in the live rules may change during the 13 weeks. A bug fix that changes decisions is logged as an amendment,
    and both versions are reported.
  - Reported monthly in EP08 onward, whatever it shows.

## H-EP08: a fine-tuned Laya brake trained on trade outcomes (meta-labeling) (written 2026-11-14, before any EP08 code on real labels was run)

EP07's general-purpose Laya brake failed: on BTC it blocked 18.6% of L1 entries, the blocked entries averaged +0.32% vs +0.07% for kept ones,
and final equity fell from 1.874x (L1) to 1.355x (L2), below the random-veto median. Live, both Laya brakes have blocked 0 of 10 entries so far.
EP08 asks one question: if Laya is fine-tuned on whether past L1 trades lost, does it pass the same brake tests EP07 failed?
The only code run before this entry is `episodes/ep08/smoke_train.py`: EP07 states with random labels, to check the training format and time.

- **Data**: EP07's L1 trades for BTCUSDT and ETHUSDT (36 months, 662 + 1,027 trades), unchanged.
  - States and the EP07 Laya answers come from `runs/ep07_laya_answers.jsonl`, joined to `episodes/ep07/l1_split.py` `l1_trades(sym)`
    on key `sym|candidate|signal_time`. The join must be one-to-one. No EP07 or live file is modified.
  - Label: `1` (should have been blocked) if the trade's net return ≤ 0, else `0`.
- **Folds (expanding window)**: test windows start 2024-11-01, 2025-05-01, 2025-11-01, 2026-05-01 (UTC), each 6 months;
  the last ends 2026-11-01 (exclusive).
  - Train = trades whose exit time is before the test start (purged). Test = trades whose signal time is inside the test window.
  - The whole test span (2024-11 … 2026-10) overlaps the period used to design the candidates (Amendment 3). This is stated in the video.
- **Models**, trained once per fold on BTC and ETH together (the state has no ticker, date or price):
  - `laya_ft`: base `convaiinnovations/laya-multilingual` rev 1720e3e3 (`vendor/laya/model`), `laya` 0.4.22,
    `laya-train --epochs 3 --freeze-encoder --seed 20261113 --device cpu`, other options at their defaults.
    Same question as EP07 ("Should this trade entry be blocked because conditions are dangerous for it?"). Veto if noul ≥ 0.5.
  - `logistic` (simple-model control): scikit-learn 1.5.2 `LogisticRegression(C=1.0, max_iter=2000)` on the same information:
    the six numeric state fields (missing → train-window median, standardized with train-window statistics) plus strategy and side as 0/1.
    It vetoes the same share of each symbol's test trades as `laya_ft` did, taking its highest probabilities.
  - `random` control: veto at `laya_ft`'s observed rate per symbol, 1,000 runs, `numpy.random.default_rng(20261113)`, BTC then ETH.
  - `laya_generic`: EP07's cached answers on the same test trades, reported for reference.
- **Measures**: per symbol, on the concatenated test trades in entry-time order. Net expectancy (fixed notional) with bootstrap CI,
  max drawdown, worst month and final equity as in EP07. Also reported: veto rate, mean net return of vetoed vs kept trades,
  and `laya_ft`'s percentile among the 1,000 random runs.
- **Hypotheses (predictions)**, per symbol:
  - (b) `laya_ft` max drawdown and worst month are smaller in size than L1's and than the random-veto medians (EP07 (b)).
  - (c) `laya_ft` net expectancy > random-veto median expectancy (EP07 (c)).
  - (e) `laya_ft` net expectancy > `logistic` net expectancy.
  - Ties count as not met. A symbol with fewer than 100 kept trades is `insufficient` and fails.
- **Verdict**: `PASS` = (b), (c) and (e) on both BTC and ETH. `WEAK` = all three on BTC only. `FAIL` otherwise. Reported whatever it shows.
- **Check, not a verdict**: re-scoring EP07's `laya_generic` over all 36 months must reproduce EP07's BTC L2 final equity (1.35492).
- **Live**: no change to the live runner. If the verdict is PASS or WEAK, adding the fine-tuned brake as a live track needs a
  separate amendment written before it runs.

## Amendments

1. **2026-11-06, after running H-EP03 / H-EP06: what "net expectancy" means.**
   - The text above does not say whether "net expectancy" is per-trade return or per-trade R.
   - The code (`protocol.trade_stats`, written before the runs) uses the **mean net return per trade at fixed notional**.
     That is the same convention as round 1, and verdicts are computed with it.
   - The two can disagree. FVG 1h, BTC-H: +0.008% per trade but −0.09R.
     Both are reported, and the video states which one a verdict uses.
   - Not re-scored; noted so the reader can judge.

2. **2026-11-06, after running H-EP02: exploratory control added.**
   - Arm `rule_randveto` vetoes the rule's entries at random, at the LLM's observed veto rate (430/573), with no stop. 20 runs per event.
   - It was added after seeing that the LLM vetoed 75% of entries, to ask whether the gain is just trading less.
   - It is labelled exploratory everywhere and does not change the H-EP02 verdicts.

3. **2026-11-07, after running H-EP07B (assembly) and before any live run or Laya re-test.**
   - **Live engine.** The live test does not use the Toobit simulator.
     - It replays the same `simulate` engine on live Binance USDT-M bars: archive files, plus the public REST klines for the last days.
     - Fills, costs and slippage are exactly the backtest's, so live vs backtest differs only by the future.
   - **Live tracks.** All tracks share one signal stream, the `L1` monthly selection.
     - Track 1, as registered: `L1` + Laya-multilingual brake + 2% risk, leverage cap 3.
     - Track 2: `L1` only, fixed notional.
     - Track 3: `L1` + Laya English (`convaiinnovations/laya`, revision pinned at download) brake, fixed notional. Same question and threshold.
     - Track 4 (exploratory, added after seeing L3): `L1` + 0.5% risk, leverage cap 3.
     - These are fixed now, before the Laya re-test is run, whatever it shows.
   - **Live objective.** The 13 weeks cannot show a +0.1%/trade edge. Each month the live per-trade mean, win rate and
     trade frequency are compared with the backtest's 95% range for the same number of trades: L1 trades over the 36 months,
     resampled 10,000 times, seed 20261107.
     - "Within range" or "outside range" is the monthly verdict.
     - Missed signals, late records and costs are reported alongside.
     - Week 13 is the first checkpoint; the run continues afterwards.
   - **Late records.** A signal recorded more than 70 minutes after its bar closed (for example, the machine was off)
     is computed by the same code but flagged `late` and reported separately.
   - **Split of the 36-month walk-forward.**
     - 2023-11 → 2024-10 was not used to design any EP03/EP06 candidate; it is reported on its own as the cleaner test.
     - 2024-11 → 2026-10 overlaps the design periods and is labelled as such.
   - **Laya re-test (backtest only).** L2 is repeated with `laya` (English) and `laya-typed-decisions`.
     - The states, question, threshold and random-veto control are unchanged.
     - All three checkpoints are reported.

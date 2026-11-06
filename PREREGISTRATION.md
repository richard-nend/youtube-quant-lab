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

## Amendments

(none)

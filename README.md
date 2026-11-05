# quant-lab

Every number quoted in the YouTube series is produced by a script in this repository.
If a number is not reproducible here, it should not be in a video.

## Quick start

```bash
uv sync
uv run pytest                                   # core engine tests
uv run python episodes/ep04/martingale_montecarlo.py
```

Market data is downloaded on first use from the public Binance archive
(<https://data.binance.vision>) and SEC EDGAR, then cached in `.cache/`.
Each script writes its results, together with the exact parameters, random seed and
raw-data URLs, to `../src/data/results/<episode>.json` (the video project that consumes them).
A snapshot of those files and the claim-by-claim ledger is kept in [`results/`](results/).

## Layout

| Path | What it is |
|---|---|
| `src/quant_lab/data/` | Binance archive loader (klines, aggTrades, open interest), SEC EDGAR loader |
| `src/quant_lab/engine/` | bar-based bracket simulator with explicit fill rules (`touch` / `through`), cost model |
| `src/quant_lab/orderflow.py` | signed volume / CVD with the correct `is_buyer_maker` sign |
| `src/quant_lab/llm.py` | minimal Gemini client (Vertex AI or AI Studio) with latency logging |
| `episodes/epXX/` | one script per claim made in episode XX |

## Rules the experiments follow

- **No look-ahead.** Signals are acted on after the bar that confirms them; market orders fill at the next bar's open.
- **Costs are always on.** Default: Binance USDT-M VIP0 (taker 0.05%, maker 0.02%, 1.5 bp slippage per taker fill).
- **Thresholds are fixed before running.** Each script's docstring states its rules; results are reported as they come out, including the ones that contradict the original script.
- **Pessimistic tie-breaks.** If stop and target fall inside the same bar, the stop is assumed first.

## Episodes

| Episode | Scripts |
|---|---|
| EP01 fake indicators | `ep01/fake_backtest_tricks.py` (Heikin-Ashi fills, repainting pivots) |
| EP02 LLM traders | `ep02/crash_event_finder.py`, `ep02/llm_crash_replay.py` |
| EP03 SMC / ICT | `ep03/smc_backtest.py` |
| EP04 martingale & grids | `ep04/martingale_montecarlo.py`, `martingale_table.py`, `blackswan_scanner.py`, `grid_bot_sim.py`, `sizing_comparison.py` |
| EP05 AI research agents | `ep05/edgar_forensics.py`, `ep05/ai_hedge_fund_probe.py` |
| EP06 order flow | `ep06/dec2024_oi_autopsy.py`, `absorption_bot_backtest.py`, `adverse_selection_ticks.py` |

Not investment advice. Simulated and historical results do not predict future results.

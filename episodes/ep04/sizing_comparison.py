"""EP04 — position sizing: martingale vs fixed fraction vs Kelly (full / half / mis-estimated).

Narration claim: "anti-martingale + Kelly make the probability of ruin 0% and are the only
mathematical answer to long-term compounding." We test what Kelly actually guarantees.

Game: even-money bet, true win probability p_true = 0.55 (edge 10%), 1,000 bets, 10,000 paths.
Strategies (fraction of CURRENT equity staked each bet unless stated):
  martingale  : $100 base on $10k, double after loss (ruin = can't post next bet)
  fixed_2pct  : 2%
  half_kelly  : f*/2 = 5%
  full_kelly  : f* = 2p-1 = 10%
  kelly_overestimated : trader believes p=0.60 -> stakes 20% (2x true Kelly)
  kelly_wrong_edge    : true p=0.50 (no edge) but trader stakes Kelly for believed p=0.55 -> 10%
Metrics: median final wealth, P(final < start), P(drawdown >= 50% at some point),
"practical ruin" = equity < 1% of start at any time.
"""

from __future__ import annotations

import numpy as np

from quant_lab.results import write_result

SEED, N_PATHS, N_BETS, P = 20261106, 10_000, 1_000, 0.55


def frac_paths(f: float, p: float, rng) -> np.ndarray:
    wins = rng.random((N_PATHS, N_BETS)) < p
    steps = np.where(wins, 1 + f, 1 - f)
    return np.cumprod(steps, axis=1)


def stats(eq: np.ndarray) -> dict:
    peak = np.maximum.accumulate(np.concatenate([np.ones((eq.shape[0], 1)), eq], axis=1), axis=1)[:, 1:]
    dd = (eq / peak - 1).min(axis=1)
    return {"median_final": float(np.median(eq[:, -1])), "p_loss": float((eq[:, -1] < 1).mean()),
            "p_drawdown_50": float((dd <= -0.5).mean()), "p_practical_ruin": float((eq.min(axis=1) < 0.01).mean())}


def martingale(p: float, rng) -> dict:
    bal = np.full(N_PATHS, 10_000.0)
    bet = np.full(N_PATHS, 100.0)
    alive = np.ones(N_PATHS, bool)
    for _ in range(N_BETS):
        w = rng.random(N_PATHS) < p
        bal[alive & w] += bet[alive & w]
        bal[alive & ~w] -= bet[alive & ~w]
        bet[alive & w] = 100
        bet[alive & ~w] *= 2
        alive &= bal >= bet
    eq = bal / 10_000
    return {"median_final": float(np.median(eq)), "p_loss": float((eq < 1).mean()),
            "p_practical_ruin": float((~alive).mean())}


SAMPLE_STEP, SAMPLE_PATHS = 5, 5


def sig4(x: float) -> float:
    return float(f"{x:.4g}")


def sampled(eq: np.ndarray) -> dict:
    """Video chart data: paths 0..4 (first five of the same seeded run, not chosen by outcome) and the
    pointwise median over all paths, every SAMPLE_STEP bets, starting at 1.0 (bet 0)."""
    idx = np.arange(0, N_BETS + 1, SAMPLE_STEP)
    full = np.concatenate([np.ones((eq.shape[0], 1)), eq], axis=1)
    peak = np.maximum.accumulate(full, axis=1)
    paths = [[sig4(v) for v in full[k, idx]] for k in range(SAMPLE_PATHS)]
    for k in range(SAMPLE_PATHS):
        assert abs(paths[k][-1] - eq[k, -1]) <= 1e-3 * abs(eq[k, -1])
    return {"bet_index": idx.tolist(), "paths": paths,
            "path_flags": [{"path": k, "min": sig4(full[k].min()), "max_drawdown": sig4((full[k] / peak[k] - 1).min()),
                            "hit_drawdown_50": bool((full[k] / peak[k] - 1).min() <= -0.5),
                            "practical_ruin": bool(full[k].min() < 0.01)} for k in range(SAMPLE_PATHS)],
            "median_pointwise": [sig4(v) for v in np.median(full[:, idx], axis=0)]}


def main():
    rng = np.random.default_rng(SEED)
    out = {"martingale": martingale(P, rng)}
    charts = {}
    for name, f, p in [("fixed_2pct", 0.02, P), ("half_kelly", 0.05, P), ("full_kelly", 0.10, P),
                       ("kelly_overestimated_2x", 0.20, P), ("kelly_wrong_edge_true_p50", 0.10, 0.50)]:
        eq = frac_paths(f, p, rng)                   # same draw order as before -> same numbers
        out[name] = stats(eq)
        if name != "kelly_wrong_edge_true_p50":
            charts[name] = sampled(eq)
            assert abs(charts[name]["median_pointwise"][-1] - out[name]["median_final"]) <= 1e-3 * out[name]["median_final"]
        del eq
    for k, v in out.items():
        print(f"{k:28} " + " ".join(f"{m}={x:.3g}" for m, x in v.items()))
    out["charts"] = {"how_chosen": "paths 0-4 of the seeded run (first five, not chosen by outcome) + pointwise median "
                                   "of all 10,000 paths; equity / start, every 5 bets",
                     **charts}
    write_result("ep04", "sizing_comparison", out,
                 params={"p_true": P, "n_paths": N_PATHS, "n_bets": N_BETS, "payoff": "even money"},
                 seed=SEED, sources=[])


if __name__ == "__main__":
    main()

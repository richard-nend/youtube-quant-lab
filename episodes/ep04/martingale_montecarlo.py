"""EP04 — Martingale ruin Monte Carlo.

Replaces the hard-coded `simCount * 0.994` in LongformEp04Components.tsx.

Model (identical to the code shown on screen, plus explicit knobs):
    balance starts at `bankroll`, base bet `base_bet`, even-money payoff.
    win  (prob p): balance += bet, bet resets to base
    loss         : balance -= bet, bet *= multiplier
    ruin         : the next required bet exceeds the remaining balance
                   (= the bot can no longer post margin for its next averaging-down step)

Outputs the headline case shown in the video and a sensitivity grid so the
script can quote whatever is TRUE instead of a convenient number.

Run:  uv run python episodes/ep04/martingale_montecarlo.py
"""

from __future__ import annotations

import numpy as np

from quant_lab.results import write_result

SEED = 20261105


def ruin_rate(p: float, multiplier: float, *, n_traders: int = 1000, n_trades: int = 500,
              bankroll: float = 10_000, base_bet: float = 100, seed: int = SEED,
              cost_per_bet: float = 0.0, return_paths: int = 0):
    """Vectorised over traders. cost_per_bet is a fraction of the bet lost to fees each bet."""
    rng = np.random.default_rng(seed)
    bal = np.full(n_traders, bankroll, dtype=float)
    bet = np.full(n_traders, base_bet, dtype=float)
    alive = np.ones(n_traders, dtype=bool)
    ruined_at = np.full(n_traders, -1)
    balance_at_ruin = np.full(n_traders, np.nan)
    peak = bal.copy()
    paths = np.empty((n_trades + 1, return_paths)) if return_paths else None
    if return_paths:
        paths[0] = bal[:return_paths]

    for t in range(n_trades):
        win = rng.random(n_traders) < p
        a = alive
        fee = bet * cost_per_bet
        bal[a & win] += bet[a & win] - fee[a & win]
        bal[a & ~win] -= bet[a & ~win] + fee[a & ~win]
        bet[a & win] = base_bet
        bet[a & ~win] *= multiplier
        peak = np.maximum(peak, bal)
        newly = a & (bal < bet)
        ruined_at[newly] = t + 1
        balance_at_ruin[newly] = bal[newly]
        alive = a & ~newly
        if return_paths:
            # after ruin the path is frozen at the balance left when the next bet became unaffordable
            paths[t + 1] = np.where(alive[:return_paths], bal[:return_paths],
                                    np.nan_to_num(balance_at_ruin[:return_paths]))

    ruined = ~alive
    out = {
        "ruin_rate": float(ruined.mean()),
        "ruined": int(ruined.sum()),
        "n_traders": n_traders,
        "median_trade_of_ruin": float(np.median(ruined_at[ruined])) if ruined.any() else None,
        "survivor_median_balance": float(np.median(bal[alive])) if alive.any() else None,
        "survivor_median_peak": float(np.median(peak[alive])) if alive.any() else None,
        "median_balance_left_at_ruin": float(np.nanmedian(balance_at_ruin)) if ruined.any() else None,
        "ruined_mask_first_paths": ruined[:return_paths].tolist() if return_paths else None,
    }
    return (out, paths) if return_paths else out


def main():
    # --- the exact setup described in the narration ---------------------------------
    headline = {f"x{m}": ruin_rate(0.60, m) for m in (1.5, 2.0)}

    # --- sensitivity grid: what assumptions WOULD be needed for ~99% ruin? -----------
    grid = []
    for p in (0.50, 0.55, 0.60):
        for m in (1.5, 2.0):
            for n in (500, 2000, 10000):
                for fee in (0.0, 0.0013):          # 0.13% round trip (taker+slippage)
                    r = ruin_rate(p, m, n_trades=n, cost_per_bet=fee)
                    grid.append({"win_rate": p, "multiplier": m, "n_trades": n,
                                 "cost_per_bet": fee, "ruin_rate": r["ruin_rate"],
                                 "median_trade_of_ruin": r["median_trade_of_ruin"]})

    # --- on-screen sample: the first 50 traders of the headline run (2.0x, 60%) --------
    # Not cherry-picked: same seed as the headline, traders 0..49. Their ruin share is
    # reported next to the paths so the chart can show it honestly.
    sample_stats, paths = ruin_rate(0.60, 2.0, return_paths=50)
    sample_paths = paths[::5].T.round(2).tolist()   # every 5th trade to keep JSON small
    sample_ruined = sample_stats["ruined_mask_first_paths"]

    for k, v in headline.items():
        print(f"headline {k}: ruin {v['ruin_rate']:.1%} ({v['ruined']}/1000), "
              f"median ruin at trade {v['median_trade_of_ruin']}")
    print("\nwin  mult  trades  fee     ruin")
    for g in grid:
        print(f"{g['win_rate']:.2f} {g['multiplier']:>4} {g['n_trades']:>7} {g['cost_per_bet']:.4f} {g['ruin_rate']:7.1%}")

    path = write_result(
        "ep04", "martingale_montecarlo",
        {"headline": headline, "grid": grid, "sample_paths_every5": sample_paths,
         "sample_paths_ruined": sample_ruined,
         "sample_ruin_share": float(np.mean(sample_ruined))},
        params={"win_rate": 0.60, "n_traders": 1000, "n_trades": 500, "bankroll": 10_000,
                "base_bet": 100, "payoff": "even money", "ruin": "next bet > balance"},
        seed=SEED, sources=[],
        note="Pure simulation, no market data. Replaces hard-coded 99.4% in the original EP04 render.",
    )
    print("\nwrote", path)


if __name__ == "__main__":
    main()

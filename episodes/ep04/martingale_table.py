"""EP04 — deterministic martingale arithmetic quoted in the narration.

Checks: "7 losses in a row -> single bet 12,800, cumulative 25,500" (long-form),
        "7 losses -> 128x" and "99 wins = $500 vs $50,000 wiped" (shorts),
        "7 opposite moves in a row < 1%".
"""

from __future__ import annotations

from quant_lab.results import write_result


def ladder(base: float, mult: float, steps: int):
    rows, cum = [], 0.0
    for k in range(steps + 1):
        bet = base * mult**k
        cum += bet
        rows.append({"losses_before": k, "bet": bet, "cumulative_staked": cum,
                     "cumulative_loss_if_this_loses": cum})
    return rows


def losses_to_wipe(base: float, mult: float, account: float) -> int:
    """Smallest number of consecutive losses after which the next bet cannot be posted."""
    k, lost = 0, 0.0
    while True:
        bet = base * mult**k
        if lost + bet > account:
            return k
        lost += bet
        k += 1


def main():
    lad = ladder(100, 2.0, 10)
    after7 = lad[7]
    probs = {f"p_loss={q}": {"7_in_a_row": q**7, "12_in_a_row": q**12} for q in (0.5, 0.4)}
    shorts = {
        "base_bet_implied_by_99_wins_500usd": 500 / 99,
        "losses_to_wipe_50k_with_that_base": losses_to_wipe(500 / 99, 2.0, 50_000),
        "multiplier_after_7_losses": 2**7,
    }
    values = {
        "ladder_base100_x2": lad,
        "after_7_losses": {"next_bet": after7["bet"],
                           "lost_so_far": lad[6]["cumulative_staked"],
                           "staked_including_8th_bet": after7["cumulative_staked"]},
        "streak_probabilities": probs,
        "shorts_consistency": shorts,
    }
    print("after 7 losses:", values["after_7_losses"])
    print("streak probs:", probs)
    print("shorts check:", shorts)
    print("wrote", write_result("ep04", "martingale_table", values,
                                params={"base": 100, "mult": 2.0}, sources=[]))


if __name__ == "__main__":
    main()

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from league.ours import brake as bk

EP09 = Path(os.environ.get("QL_EP09_RESULTS", "/Users/gimhyeong-gi/MyProjects/Youtube/trading/src/data/results/ep09.json"))
TREE_SCORES = bk.MAIN_ROOT / "runs" / "ep09" / "tree_scores.jsonl"
pytestmark = pytest.mark.skipif(not (bk.ROWS.exists() and EP09.exists() and TREE_SCORES.exists()),
                                reason=f"needs {bk.ROWS}, {EP09} and {TREE_SCORES}")
ml, mb, tb = bk.tb.ml, bk.tb.mb, bk.tb


@pytest.fixture(scope="module")
def rows():
    return bk.load_pool()


def test_brake_threshold_is_pool_q80(rows, tmp_path):
    pool = rows[rows.exit_time < ml.FOLD_STARTS[0]]
    b = bk.Brake.fit_frozen(pool, tmp_path / "brake.pkl")
    p_pool, _ = tb.oof(pool)
    want = np.quantile(tb.logistic_plus(pool, p_pool, pool, p_pool), 0.8)
    assert b.threshold == pytest.approx(want, abs=1e-12)
    b2 = bk.Brake.load(tmp_path / "brake.pkl")
    x = pool.x.iloc[0]
    assert b2.blocks(x) == b.blocks(x)
    assert b.blocks(x)[0] == (b.blocks(x)[1] >= b.threshold)


def test_brake_reproduces_ep09_logistic_plus(rows):
    """Per fold: brake fit on the purged pool, scores on the L1 test trades, top 20% vetoed per symbol and fold."""
    parts = []
    for start, end in zip(ml.FOLD_STARTS, ml.fold_ends()):
        pool = rows[rows.exit_time < start]
        test = rows[rows.is_l1 & (rows.signal_time >= start) & (rows.signal_time < end)]
        parts.append(test.assign(fold=f"{start:%Y-%m}", p=bk.Brake.fit_frozen(pool).scores(list(test.x))))
    test = pd.concat(parts, ignore_index=True)
    assert len(test) == 1156
    ref = pd.read_json(TREE_SCORES, lines=True).set_index("key").p_logit_plus
    res = json.loads(EP09.read_text())["experiments"]["tree_brake_walkforward"]["values"]
    for sym in mb.SYMS:
        t = test[test.sym == sym].sort_values("entry_time").reset_index(drop=True)
        g = t.fold.to_numpy()
        mine = mb.top_share(t.p, g, mb.VETO_SHARE)
        theirs = mb.top_share(ref.loc[t.key].to_numpy(), g, mb.VETO_SHARE)
        print(sym, "max |p - ep09|", float(np.abs(t.p - ref.loc[t.key].to_numpy()).max()),
              "blocked differ", int((mine != theirs).sum()))
        assert (mine != theirs).sum() <= 2  # ±1 trade swapped
        blocked, kept = res[sym]["blocked_vs_kept_mean_net"]["logistic_plus"]
        assert res[sym]["arms"]["logistic_plus"]["trades"] == int((~mine).sum())
        tol = 0 if (mine == theirs).all() else 2 * float(t.net_ret.abs().max()) / int(mine.sum())
        assert abs(float(t[mine].net_ret.mean()) - blocked) <= tol + 1e-12
        assert abs(float(t[~mine].net_ret.mean()) - kept) <= tol + 1e-12

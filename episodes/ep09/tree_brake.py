"""EP09 (H-EP09): a LightGBM tree on EP08b's facts, alone and as one extra paragraph in Laya's state.

Rows, folds, label and the base state come from EP08b (`runs/ep08b/rows.jsonl`, built by meta_label_b.py export).
The Laya job is the unchanged `episodes/ep08/kaggle_train_b.py`, pointed at the dataset written here.

    uv run --directory quant_lab --with scikit-learn==1.5.2 --with lightgbm==4.6.0 python episodes/ep09/tree_brake.py export
    # Kaggle: dataset runs/ep09/kaggle_data -> kernel kaggle_train_b.py -> download scores_b.jsonl into runs/ep09/
    uv run --directory quant_lab --with scikit-learn==1.5.2 --with lightgbm==4.6.0 python episodes/ep09/tree_brake.py evaluate
"""

from __future__ import annotations

import importlib.util
import json
import sys
from collections import Counter

import numpy as np
import pandas as pd

from quant_lab import LAB_ROOT


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, LAB_ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


mb = _load("meta_label_b", "episodes/ep08/meta_label_b.py")
ml = mb.ml
SEED, SYMS, VETO_SHARE = mb.SEED, mb.SYMS, mb.VETO_SHARE
RUNS = LAB_ROOT / "runs" / "ep09"
KAGGLE_DIR = RUNS / "kaggle_data"
TREE = RUNS / "tree_scores.jsonl"
LAYA_TREE = RUNS / "scores_b.jsonl"   # Kaggle kernel output, downloaded
N_BLOCKS = 5
LGB = dict(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=40, subsample=0.8, subsample_freq=1,
           colsample_bytree=0.8, reg_lambda=1.0, random_state=SEED, deterministic=True, n_jobs=1, verbose=-1)
DESC = {"long": "the trade direction", "limit_entry": "the entry order type", "stop_pct": "the stop distance",
        "target_pct": "the target distance", "reward_risk": "the reward-to-risk ratio", "cost_pct": "the round-trip cost",
        "cost_to_stop": "the cost as a share of the stop", "max_hold_h": "the maximum holding time",
        "trail12_sum_pct": "the strategy's 12-month net result", "trail12_n": "the strategy's 12-month trade count",
        "recent_win": "the strategy's recent win rate", "recent_mean_pct": "the strategy's recent average return",
        "hour": "the hour of day", "weekday": "the weekday", "ret_24h_pct": "the 24-hour return",
        "ret_7d_pct": "the 7-day return", "ret_30d_pct": "the 30-day return", "atr_pct": "the volatility",
        "atr_pctile_90d": "the volatility's 90-day percentile", "from_ma200_pct": "the distance from the 200-bar average",
        "from_20d_high_pct": "the distance from the 20-day high", "from_20d_low_pct": "the distance from the 20-day low",
        "delta_z": "the aggressive-buying z-score", "cand": "the strategy type"}


def x_frame(rows: pd.DataFrame) -> pd.DataFrame:
    x = pd.DataFrame(list(rows.x), dtype=float, index=rows.index)
    return x[sorted(x.columns)]  # fixed column order; missing stays NaN (LightGBM handles it)


def fit(train: pd.DataFrame):
    import lightgbm as lgb

    return lgb.LGBMClassifier(**LGB).fit(x_frame(train), train.label.to_numpy())


def score(model, rows: pd.DataFrame) -> tuple[np.ndarray, list[list[tuple[str, float]]]]:
    """Loss probability and the three largest contributions (strategy one-hots merged into 'cand')."""
    x = x_frame(rows)
    p = model.predict_proba(x)[:, 1]
    c = pd.DataFrame(model.predict(x, pred_contrib=True)[:, :-1], columns=x.columns)  # last column is the bias
    cand = [k for k in c.columns if k.startswith("cand_")]
    c = c.drop(columns=cand).assign(cand=c[cand].sum(axis=1))
    top = [sorted(r.items(), key=lambda kv: -abs(kv[1]))[:3] for r in c.to_dict("records")]
    return p, top


def oof(pool: pd.DataFrame) -> tuple[np.ndarray, list]:
    """Purged out-of-fold scores for the training pool: 5 contiguous time blocks."""
    order = pool.sort_values("signal_time")
    p, top = pd.Series(np.nan, index=pool.index), pd.Series(None, index=pool.index, dtype=object)
    for blk in np.array_split(order.index.to_numpy(), N_BLOCKS):
        b = pool.loc[blk]
        lo, hi = b.signal_time.min(), b.exit_time.max()
        rest = pool.drop(index=blk)
        rest = rest[(rest.exit_time < lo) | (rest.signal_time > hi)]
        pb, tb = score(fit(rest), b)
        p.loc[blk], top.loc[blk] = pb, pd.Series(tb, index=blk, dtype=object)
    assert not p.isna().any()
    return p.to_numpy(), list(top)


def paragraph(p: float, top: list[tuple[str, float]]) -> str:
    facts = ", ".join(f"{DESC[k]} ({'raises' if v > 0 else 'lowers'} it)" for k, v in top)
    return (f"A tree model trained only on earlier trades puts the chance that this trade loses money at {100 * p:.0f}%. "
            f"The facts that moved its estimate most: {facts}.")


def load_rows() -> pd.DataFrame:
    rows = pd.read_json(mb.RUNS / "rows.jsonl", lines=True)
    for c in ("signal_time", "entry_time", "exit_time"):
        rows[c] = pd.to_datetime(rows[c], utc=True)
    return rows


def logistic_plus(pool: pd.DataFrame, p_pool, test: pd.DataFrame, p_test) -> np.ndarray:
    add = lambda d, p: d.assign(x=[{**x, "tree_p": float(q)} for x, q in zip(d.x, p)])  # noqa: E731
    return mb.logistic_scores_b(add(pool, p_pool), add(test, p_test))


def export() -> None:
    rows = load_rows()
    KAGGLE_DIR.mkdir(parents=True, exist_ok=True)
    TREE.unlink(missing_ok=True)
    meta = {}
    for start, end in zip(ml.FOLD_STARTS, ml.fold_ends()):
        tag = f"{start:%Y-%m}"
        pool = rows[rows.exit_time < start]
        test = rows[rows.is_l1 & (rows.signal_time >= start) & (rows.signal_time < end)]
        assert not set(pool.key) & set(test.key)
        p_pool, top_pool = oof(pool)
        p_test, top_test = score(fit(pool), test)
        p_lp = logistic_plus(pool, p_pool, test, p_test)
        with (KAGGLE_DIR / f"train_{tag}.jsonl").open("w") as fh:
            for s, p, t, y in zip(pool.state, p_pool, top_pool, pool.label):
                fh.write(json.dumps({"state": f"{s} {paragraph(p, t)}",
                                     "questions": {"a": {"type": "noul", "instructions": mb.Q_LOSS}},
                                     "expected": {"a": int(y)}}) + "\n")
        with (KAGGLE_DIR / f"test_{tag}.jsonl").open("w") as fh:
            for k, s, p, t in zip(test.key, test.state, p_test, top_test):
                fh.write(json.dumps({"key": k, "state": f"{s} {paragraph(p, t)}"}) + "\n")
        with TREE.open("a") as fh:
            for k, p, t, q in zip(test.key, p_test, top_test, p_lp):
                fh.write(json.dumps({"key": k, "fold": tag, "p_lgbm": float(p), "p_logit_plus": float(q),
                                     "top": [k2 for k2, _ in t]}) + "\n")
        meta[tag] = {"train_rows": len(pool), "train_label1": float(pool.label.mean()), "test_rows": len(test)}
        print(tag, meta[tag], flush=True)
    (KAGGLE_DIR / "folds.json").write_text(json.dumps({"question": mb.Q_LOSS, "folds": meta}, indent=1))
    (KAGGLE_DIR / "dataset-metadata.json").write_text(json.dumps(
        {"title": "ep09-folds", "id": "richardhykim/ep09-folds", "licenses": [{"name": "CC0-1.0"}]}))
    print(open(KAGGLE_DIR / f"test_{tag}.jsonl").readline()[:1500])


def _spread(t: pd.DataFrame, col: str) -> float:
    return float(t.groupby("fold")[col].std().mean())


def evaluate() -> None:
    from sklearn.metrics import roc_auc_score

    from quant_lab.results import write_result

    asm = ml._l1_module().asm
    rows = load_rows()
    read = lambda f: pd.DataFrame([json.loads(l) for l in f.read_text().splitlines()])  # noqa: E731
    tree, lt, ft = read(TREE), read(LAYA_TREE).rename(columns={"p": "p_laya_tree"}), read(mb.SCORES).rename(columns={"p": "p_ft_b"})
    folds = {f"{s:%Y-%m}" for s in ml.FOLD_STARTS}
    for d in (tree, lt, ft):
        assert d.key.is_unique and set(d.fold) == folds, "all folds must be scored"
    test = (rows[rows.is_l1].merge(tree, on="key", validate="one_to_one")
            .merge(lt.drop(columns="fold"), on="key", validate="one_to_one")
            .merge(ft.drop(columns="fold"), on="key", validate="one_to_one"))
    assert len(test) == 1156, len(test)
    test["p_logit"] = np.nan
    for start in ml.FOLD_STARTS:
        m = test.fold == f"{start:%Y-%m}"
        test.loc[m, "p_logit"] = mb.logistic_scores_b(rows[rows.exit_time < start], test[m])
    test["cost_to_stop"] = [x["cost_to_stop"] for x in test.x]
    scorers = {"lgbm": "p_lgbm", "laya_tree": "p_laya_tree", "laya_ft_b": "p_ft_b", "logistic": "p_logit",
               "logistic_plus": "p_logit_plus", "cost_rule": "cost_to_stop"}
    rng = np.random.default_rng(SEED)
    ls = ml.loss_size
    out, st_a, st_b = {}, {}, {}
    for sym in SYMS:
        t = test[test.sym == sym].sort_values("entry_time").reset_index(drop=True)
        g = t.fold.to_numpy()
        vetoes = {k: mb.top_share(t[c], g, VETO_SHARE) for k, c in scorers.items()}
        arms = {"L1": ml._arm(t, asm), **{k: ml._arm(t[~v], asm) for k, v in vetoes.items()}}
        r = []
        for _ in range(mb.N_RANDOM):
            a = ml._arm(t[~mb.top_share(rng.random(len(t)), g, VETO_SHARE)], asm)
            r.append((a["expectancy"], a["max_dd"], a["worst_month"]))
        r = np.array(r, float)
        med = {"expectancy": float(np.median(r[:, 0])), "max_dd": float(np.median(r[:, 1])),
               "worst_month": float(np.median(r[:, 2]))}

        def b_ok(a):
            return bool(ls(a["max_dd"]) < ls(arms["L1"]["max_dd"]) and ls(a["max_dd"]) < ls(med["max_dd"])
                        and ls(a["worst_month"]) < ls(arms["L1"]["worst_month"])
                        and ls(a["worst_month"]) < ls(med["worst_month"]))

        e = {k: a["expectancy"] for k, a in arms.items()}
        a9 = {"b": b_ok(arms["lgbm"]), "c": e["lgbm"] > med["expectancy"], "e": e["lgbm"] > e["logistic"],
              "f": e["lgbm"] > e["cost_rule"]}
        gates = {"m1_spearman": float(t.p_laya_tree.corr(t.p_lgbm, method="spearman")),
                 "m2_spread": {"laya_tree": _spread(t, "p_laya_tree"), "laya_ft_b": _spread(t, "p_ft_b")}}
        taught = gates["m1_spearman"] >= 0.30 and gates["m2_spread"]["laya_tree"] > gates["m2_spread"]["laya_ft_b"]
        b9 = {"b": b_ok(arms["laya_tree"]), "c": e["laya_tree"] > med["expectancy"], "g": e["laya_tree"] > e["laya_ft_b"],
              "h": e["laya_tree"] > e["logistic_plus"], "i": e["laya_tree"] > e["lgbm"]}
        st_a[sym] = _status(arms["lgbm"]["trades"], all(a9.values()))
        st_b[sym] = "not taught" if not taught else _status(arms["laya_tree"]["trades"], all(b9.values()))
        out[sym] = {"test_trades": len(t), "arms": arms, "random_median": med, "h_ep09a": a9, "h_ep09b": b9,
                    "gates": gates, "taught": taught,
                    "random_rank_pct": {k: float((r[:, 0] < e[k]).mean()) for k in ("lgbm", "laya_tree")},
                    "auc": {k: float(roc_auc_score(t.label, t[c])) for k, c in scorers.items()},
                    "blocked_vs_kept_mean_net": {k: [float(t[v].net_ret.mean()), float(t[~v].net_ret.mean())]
                                                 for k, v in vetoes.items()},
                    "tree_top_facts": Counter(f for top in t.top for f in top).most_common(3)}
        print(sym, "exp", {k: round(v, 5) for k, v in e.items()}, "randmed", round(med["expectancy"], 5),
              "auc", {k: round(v, 3) for k, v in out[sym]["auc"].items()}, "gates", gates, a9, b9, flush=True)
    out["H-EP09a"], out["H-EP09b"] = _verdict(st_a), _verdict(st_b)
    print("H-EP09a", out["H-EP09a"], "H-EP09b", out["H-EP09b"], flush=True)
    write_result("ep09", "tree_brake_walkforward", out,
                 params={"lightgbm": "4.6.0 " + json.dumps(LGB), "oof": f"{N_BLOCKS} contiguous blocks, purged",
                         "laya": "as H-EP08b, state + tree paragraph", "veto": "top 20% per symbol and fold",
                         "n_random": mb.N_RANDOM, "preregistration": "PREREGISTRATION.md H-EP09"},
                 seed=SEED)


def _status(kept: int, ok: bool) -> str:
    return "insufficient" if kept < mb.MIN_KEPT else "met" if ok else "not met"


def _verdict(status: dict) -> dict:
    v = "PASS" if all(status[s] == "met" for s in SYMS) else "WEAK" if status["BTCUSDT"] == "met" else "FAIL"
    return {"verdict": v, "per_symbol": status}


def _selfcheck() -> None:
    """Synthetic data: OOF must be purged and the tree must find a planted non-linear signal."""
    rng = np.random.default_rng(0)
    n = 1200
    t0 = pd.Timestamp("2024-01-01", tz="UTC") + pd.to_timedelta(np.arange(n) * 6, unit="h")
    a, b_ = rng.normal(size=n), rng.normal(size=n)
    y = ((a * b_ > 0) ^ (rng.random(n) < 0.2)).astype(int)  # XOR-like: invisible to a linear model
    pool = pd.DataFrame({"signal_time": t0, "exit_time": t0 + pd.Timedelta(hours=30), "label": y,
                         "x": [{"atr_pct": float(u), "delta_z": float(v), "cand_X": 1.0} for u, v in zip(a, b_)]})
    p, top = oof(pool)
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(y, p)
    assert auc > 0.7, auc
    assert top[0][0][0] in DESC and "%" in paragraph(p[0], top[0])
    print("selfcheck ok, oof auc", round(auc, 3), paragraph(p[0], top[0]))


if __name__ == "__main__":
    {"export": export, "evaluate": evaluate, "selfcheck": _selfcheck}[sys.argv[1] if len(sys.argv) > 1 else "export"]()

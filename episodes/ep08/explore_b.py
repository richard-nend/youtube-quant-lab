"""H-EP08b follow-up, EXPLORATORY (written after the verdict; nothing here changes it).

Asks why fine-tuned Laya lost: per-fold separation, which strategies each brake blocks, what Laya's score tracks,
what the logistic weighs, and how sure we are that any AUC is above 0.5.

    uv run --directory quant_lab --with scikit-learn==1.5.2 python episodes/ep08/explore_b.py
"""

from __future__ import annotations

import importlib.util
import json

import numpy as np
import pandas as pd

from quant_lab import LAB_ROOT

_s = importlib.util.spec_from_file_location("mlb", LAB_ROOT / "episodes/ep08/meta_label_b.py")
mlb = importlib.util.module_from_spec(_s)
_s.loader.exec_module(mlb)
ml = mlb.ml
N_BOOT = 2000


def load_test() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = pd.read_json(mlb.RUNS / "rows.jsonl", lines=True)
    for c in ("signal_time", "entry_time", "exit_time"):
        rows[c] = pd.to_datetime(rows[c], utc=True)
    sc = pd.DataFrame([json.loads(l) for l in mlb.SCORES.read_text().splitlines()])
    test = rows[rows.is_l1].merge(sc, on="key", how="inner", validate="one_to_one")
    test["p_logit"] = np.nan
    for start in ml.FOLD_STARTS:
        m = test.fold == f"{start:%Y-%m}"
        test.loc[m, "p_logit"] = mlb.logistic_scores_b(rows[rows.exit_time < start], test[m])
    return rows, test


def auc_ci(y, s, rng) -> list[float]:
    from sklearn.metrics import roc_auc_score
    y, s = np.asarray(y), np.asarray(s)
    out = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            out.append(roc_auc_score(y[i], s[i]))
    return [round(float(np.quantile(out, q)), 3) for q in (0.025, 0.5, 0.975)]


def main() -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    rows, test = load_test()
    rng = np.random.default_rng(mlb.SEED)
    rep: dict = {"note": "exploratory, after the H-EP08b verdict; not a preregistered test"}

    # 1. per symbol x fold: separation and what the 20% veto removes
    per_fold = []
    for (sym, fold), t in test.groupby(["sym", "fold"]):
        g = np.zeros(len(t))
        r = {"sym": sym, "fold": fold, "n": len(t), "loss_rate": round(float(t.label.mean()), 3)}
        for arm, col in (("laya", "p"), ("logit", "p_logit")):
            v = mlb.top_share(t[col], g, mlb.VETO_SHARE)
            r[f"auc_{arm}"] = round(float(roc_auc_score(t.label, t[col])), 3)
            r[f"blocked_{arm}_pct"] = round(float(t[v].net_ret.mean() * 100), 3)
            r[f"kept_{arm}_pct"] = round(float(t[~v].net_ret.mean() * 100), 3)
        r["laya_sd"] = round(float(t.p.std()), 4)
        per_fold.append(r)
    rep["per_fold"] = per_fold

    # 2. AUC confidence intervals (bootstrap over test trades)
    rep["auc_ci95"] = {sym: {arm: auc_ci(t.label, t[col], rng) for arm, col in (("laya", "p"), ("logit", "p_logit"))}
                       for sym, t in test.groupby("sym")}

    # 3. which strategies each brake blocks (share of each candidate's trades vetoed)
    blocked = {}
    for sym, t in test.groupby("sym"):
        g = t.fold.to_numpy()
        vl, vg = mlb.top_share(t.p, g, mlb.VETO_SHARE), mlb.top_share(t.p_logit, g, mlb.VETO_SHARE)
        blocked[sym] = {c: {"n": int((t.candidate == c).sum()),
                            "laya_pct": round(float(vl[(t.candidate == c).to_numpy()].mean() * 100), 1),
                            "logit_pct": round(float(vg[(t.candidate == c).to_numpy()].mean() * 100), 1),
                            "loss_rate": round(float(t[t.candidate == c].label.mean()), 3)}
                        for c in sorted(t.candidate.unique())}
    rep["blocked_by_candidate"] = blocked

    # 4. what Laya's score tracks: rank correlation with each ex-ante feature on test trades
    xt, _ = mlb._x_frame(test.x)
    corr = {c: float(pd.Series(test.p.to_numpy()).corr(xt[c].reset_index(drop=True), method="spearman"))
            for c in xt.columns if xt[c].nunique() > 1}
    rep["laya_score_spearman_top"] = {k: round(v, 3) for k, v in sorted(corr.items(), key=lambda kv: -abs(kv[1]))[:8]}

    # 5. what the logistic weighs (last fold's model, standardized coefficients; + = more likely to lose)
    tr = rows[rows.exit_time < ml.FOLD_STARTS[-1]]
    xtr, _ = mlb._x_frame(tr.x)
    m = LogisticRegression(C=1.0, max_iter=2000).fit(StandardScaler().fit_transform(xtr), tr.label.to_numpy())
    coef = dict(zip(xtr.columns, m.coef_[0]))
    rep["logit_coef_last_fold_top"] = {k: round(float(v), 3) for k, v in sorted(coef.items(), key=lambda kv: -abs(kv[1]))[:8]}

    from quant_lab.results import write_result
    write_result("ep08", "meta_label_explore_b", rep,
                 params={"status": "exploratory, written after the H-EP08b verdict", "n_boot": N_BOOT}, seed=mlb.SEED)
    out = mlb.RUNS / "explore_b.json"
    out.write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    print(json.dumps(rep, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()

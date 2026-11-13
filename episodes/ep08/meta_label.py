"""EP08 (H-EP08): fine-tuned Laya brake trained on L1 trade outcomes (meta-labeling).

Dataset = EP07's Laya answer cache (every L1 trade's state + generic q1) joined to EP07 L1 trades.
No EP07 or live file is modified; EP07 modules are only imported.

    uv run --directory quant_lab python episodes/ep08/meta_label.py data
"""

from __future__ import annotations

import importlib.util
import json
import sys

import pandas as pd

from quant_lab import LAB_ROOT

SYMS = ("BTCUSDT", "ETHUSDT")
SEED = 20261113
CACHE_EP07 = LAB_ROOT / "runs" / "ep07_laya_answers.jsonl"
FOLD_STARTS = [pd.Timestamp(d, tz="UTC") for d in ("2024-11-01", "2025-05-01", "2025-11-01", "2026-05-01")]
FOLD_END_LAST = pd.Timestamp("2026-11-01", tz="UTC")  # exclusive


def label_of(net_ret: float) -> int:
    """1 = should have been blocked (lost or flat after costs)."""
    return int(net_ret <= 0)


def fold_ends() -> list[pd.Timestamp]:
    return FOLD_STARTS[1:] + [FOLD_END_LAST]


def split(rows: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(train, test). Train is purged: only trades already closed before the test starts."""
    train = rows[rows.exit_time < start]
    test = rows[(rows.signal_time >= start) & (rows.signal_time < end)]
    return train, test


def _l1_module():
    spec = importlib.util.spec_from_file_location("l1_split", LAB_ROOT / "episodes/ep07/l1_split.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_rows() -> pd.DataFrame:
    """One row per EP07 L1 trade: key, sym, candidate, times, net_ret, state, q1_generic, label."""
    cache = []
    for line in CACHE_EP07.read_text().splitlines():
        r = json.loads(line)
        sym, cand, sig = r["key"].split("|")
        cache.append({"key": r["key"], "sym": sym, "candidate": cand, "signal_time": pd.Timestamp(sig),
                      "state": r["state"], "q1_generic": r["q1"]})
    cache = pd.DataFrame(cache)
    l1 = _l1_module()
    parts = []
    for sym in SYMS:
        t = l1.l1_trades(sym)[["signal_time", "entry_time", "exit_time", "gross_ret", "net_ret"]]
        c = cache[cache.sym == sym]
        # L1 trades one candidate per month, so signal_time is unique within a symbol
        assert t.signal_time.is_unique and c.signal_time.is_unique, sym
        j = c.merge(t, on="signal_time", how="inner", validate="one_to_one")
        assert len(j) == len(t) == len(c), (sym, len(j), len(t), len(c))
        parts.append(j)
    rows = pd.concat(parts, ignore_index=True)
    rows["label"] = rows.net_ret.map(label_of)
    return rows.sort_values(["sym", "entry_time"]).reset_index(drop=True)


NUMERIC = ("ret_24h_pct", "ret_7d_pct", "atr_pct", "delta_z", "from_20d_high_pct", "from_20d_low_pct")
Q1 = "Should this trade entry be blocked because conditions are dangerous for it?"  # = EP07 assembly_walkforward.Q1
LAYA_BASE = LAB_ROOT / "vendor" / "laya" / "model"
RUNS = LAB_ROOT / "runs" / "ep08"
SCORES = RUNS / "scores.jsonl"
# shortcut: caches are keyed by fold and trade key only; if the dataset or training options change, delete runs/ep08/


def _features(states: pd.Series, med: pd.Series | None = None):
    x = pd.DataFrame([{k: s.get(k) for k in NUMERIC} for s in states], dtype=float)
    med = x.median() if med is None else med
    x = x.fillna(med).fillna(0.0)
    x["is_breakout"] = [float(s["strategy"].startswith("breakout")) for s in states]
    x["is_long"] = [float(s["side"] == "long") for s in states]
    return x, med


def logistic_scores(train: pd.DataFrame, test: pd.DataFrame):
    """p(label=1) for each test row. Imputation and scaling use train-window statistics only."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    xtr, med = _features(train.state)
    xte, _ = _features(test.state, med)
    sc = StandardScaler().fit(xtr)
    model = LogisticRegression(C=1.0, max_iter=2000).fit(sc.transform(xtr), train.label.to_numpy())
    return model.predict_proba(sc.transform(xte))[:, 1]


def laya_ft_scores(fold_start: pd.Timestamp, train: pd.DataFrame, test: pd.DataFrame) -> dict[str, float]:
    """Fine-tune Laya on this fold's train rows (once, cached) and score its test rows. Returns key -> p(label=1)."""
    import os
    import subprocess
    import time

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    fold = RUNS / f"fold_{fold_start:%Y-%m}"
    ckpt = fold / "ckpt"
    if not (ckpt / "config.json").exists():
        fold.mkdir(parents=True, exist_ok=True)
        data = fold / "train.jsonl"
        with data.open("w") as fh:
            for s, y in zip(train.state, train.label):
                fh.write(json.dumps({"state": s, "questions": {"a": {"type": "noul", "instructions": Q1}},
                                     "expected": {"a": int(y)}}) + "\n")
        cmd = ["laya-train", "--data", str(data), "--base", str(LAYA_BASE), "--out", str(ckpt), "--epochs", "3",
               "--freeze-encoder", "--seed", str(SEED), "--device", "cpu"]
        t0 = time.time()
        p = subprocess.run(cmd, capture_output=True, text=True)
        (fold / "train_log.txt").write_text(p.stdout + p.stderr)
        if p.returncode:
            sys.exit(f"laya-train failed for {fold.name}; see {fold / 'train_log.txt'}")
        log = json.loads((RUNS / "train_log.json").read_text()) if (RUNS / "train_log.json").exists() else {}
        log[fold.name] = {"train_rows": len(train), "label1_share": float(train.label.mean()), "secs": round(time.time() - t0, 1)}
        (RUNS / "train_log.json").write_text(json.dumps(log, indent=1))
    import laya
    agent = laya.load(str(ckpt), device="cpu")
    return {k: float(agent.predict(s, {"a": {"type": "noul", "instructions": Q1}})["answers"]["a"]["noul"])
            for k, s in zip(test.key, test.state)}


def train_all() -> None:
    rows = load_rows()
    done = set()
    if SCORES.exists():
        done = {json.loads(l)["fold"] for l in SCORES.read_text().splitlines()}
    RUNS.mkdir(parents=True, exist_ok=True)
    for start, end in zip(FOLD_STARTS, fold_ends()):
        if f"{start:%Y-%m}" in done:
            continue
        train, test = split(rows, start, end)
        assert not set(train.key) & set(test.key)
        p_logit = logistic_scores(train, test)
        p_ft = laya_ft_scores(start, train, test)
        with SCORES.open("a") as fh:
            for k, pl in zip(test.key, p_logit):
                fh.write(json.dumps({"key": k, "fold": f"{start:%Y-%m}", "p_ft": p_ft[k], "p_logit": float(pl)}) + "\n")
        print(start.date(), "train", len(train), "test", len(test), flush=True)


N_RANDOM, MIN_KEPT = 1_000, 100


def loss_size(x: float) -> float:
    """Size of a loss (drawdown or worst month); a gain counts as no loss."""
    return max(0.0, -float(x))


def verdict(per_sym: dict[str, dict]) -> dict:
    """per_sym[sym] = {"b": bool, "c": bool, "e": bool, "kept_trades": int} -> H-EP08 verdict."""
    status = {}
    for sym, r in per_sym.items():
        if r["kept_trades"] < MIN_KEPT:
            status[sym] = "insufficient"
        else:
            status[sym] = "met" if (r["b"] and r["c"] and r["e"]) else "not met"
    if all(status[s] == "met" for s in SYMS):
        v = "PASS"
    elif status["BTCUSDT"] == "met":
        v = "WEAK"
    else:
        v = "FAIL"
    return {"verdict": v, "per_symbol": status}


def _arm(t: pd.DataFrame, asm) -> dict:
    from quant_lab.protocol import trade_stats

    st = trade_stats(t)
    cs = asm.curve_stats(t)
    return {"trades": st.get("trades", 0), "win_rate": st.get("win_rate"), "expectancy": st.get("expectancy"),
            "ci95": st.get("ci95"), "final": cs["final"], "max_dd": cs["max_dd"], "worst_month": cs["worst_month"],
            "curve": cs.get("curve", [])}


def evaluate() -> None:
    import numpy as np

    from quant_lab.results import write_result

    asm = _l1_module().asm
    rows = load_rows()
    scores = pd.DataFrame([json.loads(l) for l in SCORES.read_text().splitlines()])
    assert scores.key.is_unique and set(scores.fold) == {f"{s:%Y-%m}" for s in FOLD_STARTS}, "train all folds first"
    rng = np.random.default_rng(SEED)
    out, per_sym = {}, {}
    for sym in SYMS:
        t = rows[rows.sym == sym].merge(scores, on="key", how="inner", validate="one_to_one")
        t = t.sort_values("entry_time").reset_index(drop=True)
        n = len(t)
        veto_ft = (t.p_ft >= 0.5).to_numpy()
        veto_gen = (t.q1_generic >= 0.5).to_numpy()
        rate = float(veto_ft.mean())
        k = int(round(rate * n))
        veto_log = np.zeros(n, bool)
        veto_log[np.argsort(-t.p_logit.to_numpy(), kind="stable")[:k]] = True
        arms = {"L1": _arm(t, asm), "laya_generic": _arm(t[~veto_gen], asm),
                "laya_ft": _arm(t[~veto_ft], asm), "logistic": _arm(t[~veto_log], asm)}
        rand = []
        for _ in range(N_RANDOM):
            keep = rng.random(n) >= rate
            a = _arm(t[keep], asm)
            rand.append({"expectancy": a["expectancy"], "max_dd": a["max_dd"], "worst_month": a["worst_month"]})
        r_exp = np.array([r["expectancy"] if r["expectancy"] is not None else np.nan for r in rand])
        med = {"expectancy": float(np.nanmedian(r_exp)),
               "max_dd": float(np.median([r["max_dd"] for r in rand])),
               "worst_month": float(np.median([r["worst_month"] for r in rand]))}
        ft = arms["laya_ft"]
        exp_ft = ft["expectancy"] if ft["expectancy"] is not None else -np.inf
        exp_log = arms["logistic"]["expectancy"] if arms["logistic"]["expectancy"] is not None else -np.inf
        checks = {
            "b": bool(loss_size(ft["max_dd"]) < loss_size(arms["L1"]["max_dd"])
                      and loss_size(ft["max_dd"]) < loss_size(med["max_dd"])
                      and loss_size(ft["worst_month"]) < loss_size(arms["L1"]["worst_month"])
                      and loss_size(ft["worst_month"]) < loss_size(med["worst_month"])),
            "c": bool(exp_ft > med["expectancy"]),
            "e": bool(exp_ft > exp_log),
        }
        per_sym[sym] = {**checks, "kept_trades": ft["trades"]}
        out[sym] = {"test_trades": n, "veto_rate_ft": rate, "veto_rate_generic": float(veto_gen.mean()),
                    "veto_count_logistic": k, "arms": arms, "random_median": med, "checks": checks,
                    "random_rank_pct": float(np.nanmean(r_exp < exp_ft)),
                    "blocked_mean_net": float(t[veto_ft].net_ret.mean()) if veto_ft.any() else None,
                    "kept_mean_net": float(t[~veto_ft].net_ret.mean()) if (~veto_ft).any() else None,
                    "label1_share_test": float(t.label.mean()),
                    "per_fold": {f: {"trades": int((t.fold == f).sum()), "veto_ft": int(veto_ft[t.fold == f].sum())}
                                 for f in sorted(t.fold.unique())}}
        print(sym, "n", n, "veto_ft", round(rate, 3), "exp L1/ft/log/randmed",
              [None if v is None else round(v, 5) for v in (arms["L1"]["expectancy"], ft["expectancy"],
                                                            arms["logistic"]["expectancy"], med["expectancy"])],
              checks, flush=True)
    out.update(verdict(per_sym))

    # reproduction check (not a verdict): EP07 generic brake over all 36 months, BTC
    b = rows[rows.sym == "BTCUSDT"].sort_values("entry_time")
    rep = asm.curve_stats(b[b.q1_generic < 0.5])["final"]
    out["reproduction_check"] = {"btc_l2_final_36m": rep, "ep07_reported": 1.35492, "match": abs(rep - 1.35492) < 1e-4}
    print("verdict", out["verdict"], out["per_symbol"], "reproduction", out["reproduction_check"], flush=True)
    write_result("ep08", "meta_label_walkforward", out,
                 params={"folds": [f"{s:%Y-%m-%d}" for s in FOLD_STARTS], "fold_end_last": f"{FOLD_END_LAST:%Y-%m-%d}",
                         "laya": "convaiinnovations/laya-multilingual@1720e3e3 fine-tuned per fold, laya 0.4.22, "
                                 "laya-train --epochs 3 --freeze-encoder --seed 20261113 --device cpu",
                         "logistic": "scikit-learn 1.5.2 LogisticRegression(C=1.0, max_iter=2000), matched veto count",
                         "q1": Q1, "veto_threshold": 0.5, "n_random": N_RANDOM, "min_kept_trades": MIN_KEPT,
                         "preregistration": "PREREGISTRATION.md H-EP08 (commit 984ff98)"},
                 seed=SEED)


def main(cmd: str) -> None:
    if cmd == "train":
        train_all()
        return
    if cmd == "evaluate":
        evaluate()
        return
    if cmd == "data":
        rows = load_rows()
        print({s: len(rows[rows.sym == s]) for s in SYMS}, "total", len(rows))
        for start, end in zip(FOLD_STARTS, fold_ends()):
            train, test = split(rows, start, end)
            assert not set(train.key) & set(test.key)
            print(start.date(), "train", len(train), "test", {s: int((test.sym == s).sum()) for s in SYMS})
    else:
        sys.exit(f"unknown command {cmd}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data")

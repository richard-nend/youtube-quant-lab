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
        t = l1.l1_trades(sym)[["signal_time", "entry_time", "exit_time", "net_ret"]]
        c = cache[cache.sym == sym]
        # L1 trades one candidate per month, so signal_time is unique within a symbol
        assert t.signal_time.is_unique and c.signal_time.is_unique, sym
        j = c.merge(t, on="signal_time", how="inner", validate="one_to_one")
        assert len(j) == len(t) == len(c), (sym, len(j), len(t), len(c))
        parts.append(j)
    rows = pd.concat(parts, ignore_index=True)
    rows["label"] = rows.net_ret.map(label_of)
    return rows.sort_values(["sym", "entry_time"]).reset_index(drop=True)


def main(cmd: str) -> None:
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

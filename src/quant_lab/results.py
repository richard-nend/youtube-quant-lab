"""Write experiment outputs to src/data/results/<ep>.json for the Remotion components.

Every entry carries provenance: which script, parameters, seed, raw data URLs and
when it ran. A number without an entry here must not appear in narration.
"""

from __future__ import annotations

import fcntl
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from quant_lab import RESULTS_DIR, PROJECT_ROOT
from quant_lab.data import binance


def _clean(x):
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, np.generic):
        x = x.item()
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
    return x


def ohlc_rows(b, s: int, e: int, nd: int = 2, extra: dict | None = None) -> list[dict]:
    """Bars b.iloc[s:e] as compact chart rows {t,o,h,l,c[,extra...]} (video charts only).

    extra: name -> array aligned with b (e.g. per-bar taker delta), rounded to nd decimals.
    """
    s, e = max(int(s), 0), min(int(e), len(b))
    rows = []
    for i in range(s, e):
        r = b.iloc[i]
        row = {"t": b.index[i].isoformat(), "o": round(float(r.open), nd), "h": round(float(r.high), nd),
               "l": round(float(r.low), nd), "c": round(float(r.close), nd)}
        for k, arr in (extra or {}).items():
            row[k] = round(float(arr[i]), nd)
        rows.append(row)
    return rows


def write_result(episode: str, key: str, values: dict, *, params: dict, seed: int | None = None,
                 sources: list[str] | None = None, note: str = "") -> Path:
    path = RESULTS_DIR / f"{episode}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    # two experiments of one episode can finish at once: lock the read-modify-write, then replace atomically
    with open(path.with_suffix(".lock"), "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        doc = json.loads(path.read_text()) if path.exists() else {"episode": episode, "experiments": {}}
        _merge(doc, key, values, params, seed, sources, note)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n")
        os.replace(tmp, path)
    return path


def _merge(doc, key, values, params, seed, sources, note):
    script = Path(sys.argv[0]).resolve()
    try:
        script_rel = str(script.relative_to(PROJECT_ROOT))
    except ValueError:
        script_rel = str(script)
    doc["experiments"][key] = _clean({
        "values": values,
        "provenance": {
            "script": script_rel,
            "params": params,
            "seed": seed,
            "data_sources": sources if sources is not None else list(binance.SOURCES),
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "note": note,
        },
    })


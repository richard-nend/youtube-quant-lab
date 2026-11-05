"""Write experiment outputs to src/data/results/<ep>.json for the Remotion components.

Every entry carries provenance: which script, parameters, seed, raw data URLs and
when it ran. A number without an entry here must not appear in narration.
"""

from __future__ import annotations

import json
import math
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


def write_result(episode: str, key: str, values: dict, *, params: dict, seed: int | None = None,
                 sources: list[str] | None = None, note: str = "") -> Path:
    path = RESULTS_DIR / f"{episode}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = json.loads(path.read_text()) if path.exists() else {"episode": episode, "experiments": {}}
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
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n")
    return path

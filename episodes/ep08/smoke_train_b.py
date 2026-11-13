"""EP08b pre-registration timing check: English checkpoint, full fine-tune (encoder NOT frozen).

Random labels on EP07 cached states (no trade outcome read). Also counts L0 trades per symbol to size training.

    uv run --directory quant_lab --with vendor/laya/laya-0.4.22-py3-none-any.whl python episodes/ep08/smoke_train_b.py
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import time

from quant_lab import LAB_ROOT

_s = importlib.util.spec_from_file_location("smoke", LAB_ROOT / "episodes/ep08/smoke_train.py")
smoke = importlib.util.module_from_spec(_s)
_s.loader.exec_module(smoke)

BASE_EN = LAB_ROOT / "vendor" / "laya" / "model-en"
OUT = LAB_ROOT / "runs" / "ep08b" / "smoke"


def train(data, out, device: str) -> tuple[int, float, str]:
    cmd = ["laya-train", "--data", str(data), "--base", str(BASE_EN), "--out", str(out), "--epochs", "1",
           "--seed", "20261113", "--device", device]
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True)
    return p.returncode, time.time() - t0, (p.stdout + p.stderr)[-1200:]


def l0_counts() -> dict:
    import pandas as pd
    from quant_lab.data import binance
    from quant_lab.engine.backtest import simulate

    l1 = importlib.util.spec_from_file_location("l1_split", LAB_ROOT / "episodes/ep07/l1_split.py")
    mod = importlib.util.module_from_spec(l1)
    l1.loader.exec_module(mod)
    asm = mod.asm
    out = {}
    for sym in ("BTCUSDT", "ETHUSDT"):
        bars = {tf: binance.klines(sym, tf, asm.START, asm.END, market="um") for tf in ("1h", "4h")}
        for name, (tf, zwin, fn, sim) in asm.CANDIDATES.items():
            b = bars[tf]
            t = simulate(b, fn(b) if fn else asm.flow_signals(b, zwin), asm.COSTS, **sim)
            out[f"{sym}|{name}"] = {"all": len(t), "exit_before_2026-05": int((t.exit_time < pd.Timestamp("2026-05-01", tz="UTC")).sum())}
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    report = {"l0": l0_counts()}
    report["l0_train_rows_last_fold"] = sum(v["exit_before_2026-05"] for v in report["l0"].values())
    data = OUT / "two_hundred.jsonl"
    smoke.write_jsonl(data, smoke.states(200), as_float=False)
    for dev in ("mps", "cpu"):
        code, secs, log = train(data, OUT / f"ckpt_{dev}", dev)
        report[f"t200_{dev}"] = {"rc": code, "secs": round(secs, 1), "log_tail": log}
        if code == 0:
            break  # one working device is enough for the estimate
    (OUT / "report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "l0"}, indent=1)[:3000])
    print("l0 per candidate:", {k: v["all"] for k, v in report["l0"].items()})


if __name__ == "__main__":
    main()

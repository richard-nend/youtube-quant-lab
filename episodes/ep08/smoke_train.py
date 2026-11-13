"""EP08 Task 1: minimum run check of `laya-train` before H-EP08 is registered.

Uses EP07 cached states with RANDOM labels (no trade outcome is read), so nothing here can leak results.
Finds: the accepted `expected` format, that a fine-tuned checkpoint loads and predicts, and train seconds per row-epoch.

    uv run --directory quant_lab --with vendor/laya/laya-0.4.22-py3-none-any.whl python episodes/ep08/smoke_train.py
"""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import time

from quant_lab import LAB_ROOT

Q1 = "Should this trade entry be blocked because conditions are dangerous for it?"  # = assembly_walkforward.Q1
BASE = LAB_ROOT / "vendor" / "laya" / "model"
OUT = LAB_ROOT / "runs" / "ep08" / "smoke"
CACHE = LAB_ROOT / "runs" / "ep07_laya_answers.jsonl"


def states(n: int) -> list:
    with CACHE.open() as fh:
        return [json.loads(next(fh))["state"] for _ in range(n)]


def write_jsonl(path, rows, as_float: bool) -> None:
    rng = random.Random(20261113)
    with path.open("w") as fh:
        for s in rows:
            y = rng.randint(0, 1)
            fh.write(json.dumps({"state": s, "questions": {"a": {"type": "noul", "instructions": Q1}},
                                 "expected": {"a": float(y) if as_float else y}}) + "\n")


def train(data, out, epochs: int, device: str) -> tuple[int, float, str]:
    cmd = ["laya-train", "--data", str(data), "--base", str(BASE), "--out", str(out), "--epochs", str(epochs),
           "--freeze-encoder", "--seed", "20261113", "--device", device]
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True)
    return p.returncode, time.time() - t0, (p.stdout + p.stderr)[-1500:]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    report = {}

    # 1) format: int first, float if rejected
    fmt = None
    for as_float in (False, True):
        data = OUT / f"two_{'float' if as_float else 'int'}.jsonl"
        write_jsonl(data, states(2), as_float)
        code, secs, log = train(data, OUT / "ckpt2", 1, "cpu")
        report[f"two_rows_{'float' if as_float else 'int'}"] = {"rc": code, "secs": round(secs, 1), "log_tail": log}
        if code == 0:
            fmt = "float" if as_float else "int"
            break
    report["expected_format"] = fmt
    if fmt is None:
        print(json.dumps(report, indent=1, ensure_ascii=False))
        sys.exit("laya-train rejected both formats")

    # 2) load + predict
    import laya
    agent = laya.load(str(OUT / "ckpt2"), device="cpu")
    p = agent.predict(states(1)[0], {"a": {"type": "noul", "instructions": Q1}})["answers"]["a"]["noul"]
    assert 0.0 <= float(p) <= 1.0, p
    report["predict_ok"] = float(p)

    # 3) timing: 200 rows x 1 epoch on cpu and mps
    data = OUT / "two_hundred.jsonl"
    write_jsonl(data, states(200), fmt == "float")
    for dev in ("cpu", "mps"):
        code, secs, log = train(data, OUT / f"ckpt200_{dev}", 1, dev)
        report[f"t200_{dev}"] = {"rc": code, "secs": round(secs, 1),
                                 "sec_per_row_epoch": round(secs / 200, 3) if code == 0 else None,
                                 "log_tail": log if code else ""}
    (OUT / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(json.dumps({k: (v if not isinstance(v, dict) else {a: b for a, b in v.items() if a != "log_tail"})
                      for k, v in report.items()}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()

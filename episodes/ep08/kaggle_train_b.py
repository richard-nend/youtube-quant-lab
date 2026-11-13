"""H-EP08b Kaggle job: full fine-tune of Laya (English checkpoint) per fold, then score that fold's L1 test trades.

Runs as a private Kaggle script kernel (GPU T4 x2, internet on) with the private dataset `ep08b-folds`, which holds
train_<fold>.jsonl, test_<fold>.jsonl, folds.json and the laya 0.4.22 wheel (all built by meta_label_b.py export).
Two folds train at a time, one per GPU. Output: /kaggle/working/scores_b.jsonl (key, fold, p) + train_<fold>.log.
"""

import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

IN = (glob.glob("/kaggle/input/*/folds.json") or glob.glob("/kaggle/input/**/folds.json", recursive=True))[0].rsplit("/", 1)[0]
WORK = "/kaggle/working"
BASE = "/kaggle/temp/model-en"
REPO, REV, SHA16 = "convaiinnovations/laya", "7b928d82", "891102d372688fc2"
SEED, EPOCHS = 20261113, 4
cfg = json.load(open(f"{IN}/folds.json"))
Q = cfg["question"]
FOLDS = sorted(cfg["folds"])


def sh(cmd):
    print("$", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)


sh(f"pip install -q {glob.glob(IN + '/laya-*.whl')[0]}")
from huggingface_hub import snapshot_download  # noqa: E402

snapshot_download(REPO, revision=REV, local_dir=BASE, ignore_patterns=["multilingual/*", "typed-decisions/*", "eval/*"])
h = hashlib.sha256()
with open(f"{BASE}/model.safetensors", "rb") as f:
    for b in iter(lambda: f.read(1 << 24), b""):
        h.update(b)
assert h.hexdigest().startswith(SHA16), "checkpoint differs from the registered one"


def train(fold, gpu):
    out = f"/kaggle/temp/ckpt_{fold}"
    log = open(f"{WORK}/train_{fold}.log", "w")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    cmd = ["laya-train", "--data", f"{IN}/train_{fold}.jsonl", "--base", BASE, "--out", out,
           "--epochs", str(EPOCHS), "--seed", str(SEED), "--device", "cuda"]
    return subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env), out


def score(fold, ckpt):
    import laya
    agent = laya.load(ckpt, device="cuda")
    rows = [json.loads(l) for l in open(f"{IN}/test_{fold}.jsonl")]
    qs = {"a": {"type": "noul", "instructions": Q}}
    res = agent.predict_batch([r["state"] for r in rows], qs, batch_size=32)
    with open(f"{WORK}/scores_b.jsonl", "a") as fh:
        for r, x in zip(rows, res):
            fh.write(json.dumps({"key": r["key"], "fold": fold, "p": float(x["answers"]["a"]["noul"])}) + "\n")
    del agent


t0 = time.time()
for pair in (FOLDS[0:2], FOLDS[2:4]):
    jobs = [(f, *train(f, g)) for g, f in enumerate(pair)]
    for f, p, out in jobs:
        rc = p.wait()
        print(f, "train rc", rc, "elapsed", round(time.time() - t0), "s", flush=True)
        if rc:
            print(open(f"{WORK}/train_{f}.log").read()[-3000:])
            sys.exit(f"training failed for fold {f}")
    for f, _, out in jobs:
        score(f, out)
        shutil.rmtree(out, ignore_errors=True)
        print(f, "scored", flush=True)
print("done", round(time.time() - t0), "s")

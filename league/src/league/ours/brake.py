"""우리 팀 브레이크: EP09 logistic_plus (EP08b 사실 + LightGBM 트리 확률을 한 열로 더한 로지스틱), 시즌 시작에 1회 동결.

    uv run --project league python -m league.ours.brake   # EP08b L0 풀 전체로 적합 -> league/config/brake.pkl
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from .signals import MAIN_ROOT, _load

tb = _load("tree_brake", "episodes/ep09/tree_brake.py")
ROWS = MAIN_ROOT / "runs" / "ep08b" / "rows.jsonl"
PATH = Path(__file__).resolve().parents[3] / "config" / "brake.pkl"
Q = 0.8  # EP09 veto = top 20%


def load_pool() -> pd.DataFrame:
    """EP08b L0 rows (tree_brake.load_rows, read from QL_MAIN_ROOT)."""
    rows = pd.read_json(ROWS, lines=True)
    for c in ("signal_time", "entry_time", "exit_time"):
        rows[c] = pd.to_datetime(rows[c], utc=True)
    return rows


META = PATH.with_name("brake_meta.json")


def write_meta(pkl: Path, threshold: float, pool_rows: int, meta: Path = META) -> Path:
    """git에 올리는 동결 기록: 임계값, 적합 시각, 라이브러리 판본, pkl sha256(pkl 자체는 pickle이라 판본이 바뀌면 못 읽을 수 있다)."""
    import hashlib
    import json
    import platform
    from datetime import datetime, timezone
    from importlib.metadata import version

    meta.write_text(json.dumps({
        "threshold": threshold, "quantile": Q, "pool_rows": pool_rows,
        "fit_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sha256": hashlib.sha256(Path(pkl).read_bytes()).hexdigest(),
        "versions": {"python": platform.python_version(),
                     **{k: version(k) for k in ("numpy", "pandas", "scikit-learn", "lightgbm")}},
    }, indent=1) + "\n")
    return meta


class Brake:
    def __init__(self, tree, med, scaler, logit, cols, threshold):
        self.tree, self.med, self.scaler, self.logit, self.cols, self.threshold = tree, med, scaler, logit, cols, threshold

    @classmethod
    def fit_frozen(cls, pool: pd.DataFrame, path: Path | None = None) -> Brake:
        """tree_brake.logistic_plus의 적합 절반: 로지스틱은 OOF 트리 확률로, 새 매매의 트리 확률은 풀 전체 트리로."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        p_pool, _ = tb.oof(pool)
        xtr, med = tb.mb._x_frame([{**x, "tree_p": float(q)} for x, q in zip(pool.x, p_pool)])
        sc = StandardScaler().fit(xtr)
        m = LogisticRegression(C=1.0, max_iter=2000).fit(sc.transform(xtr), pool.label.to_numpy())
        thr = float(np.quantile(m.predict_proba(sc.transform(xtr))[:, 1], Q))
        b = cls(tb.fit(pool), med, sc, m, list(xtr.columns), thr)
        if path is not None:
            Path(path).write_bytes(pickle.dumps(b))
        return b

    @staticmethod
    def load(path: Path = PATH) -> Brake:
        return pickle.loads(Path(path).read_bytes())  # 우리가 쓴 파일만 읽는다

    def scores(self, xs: list[dict]) -> np.ndarray:
        p = self.tree.predict_proba(tb.x_frame(pd.DataFrame({"x": xs}))[self.tree.feature_name_])[:, 1]
        xte, _ = tb.mb._x_frame([{**x, "tree_p": float(q)} for x, q in zip(xs, p)], self.med)
        return self.logit.predict_proba(self.scaler.transform(xte[self.cols]))[:, 1]

    def blocks(self, facts: dict) -> tuple[bool, float]:
        s = float(self.scores([facts])[0])
        return s >= self.threshold, s


if __name__ == "__main__":
    # -m 으로 돌면 이 파일은 __main__ 이라 pickle 이 __main__.Brake 를 가리킨다 → 데몬에서 못 읽음. 패키지 쪽 클래스로 저장.
    from league.ours.brake import Brake as PkgBrake, load_pool as pkg_pool, write_meta as pkg_meta
    pool = pkg_pool()
    thr = PkgBrake.fit_frozen(pool, PATH).threshold
    print("threshold", thr, "->", PATH, pkg_meta(PATH, thr, len(pool)))

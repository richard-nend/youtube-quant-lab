import hashlib
import json

from league.ours import brake as bk


def test_brake_meta_records_threshold_versions_and_hash(tmp_path):
    pkl = tmp_path / "brake.pkl"
    pkl.write_bytes(b"frozen")
    meta = bk.write_meta(pkl, 0.61, 1234, tmp_path / "brake_meta.json")
    d = json.loads(meta.read_text())
    assert d["threshold"] == 0.61 and d["pool_rows"] == 1234 and d["quantile"] == bk.Q
    assert d["sha256"] == hashlib.sha256(b"frozen").hexdigest() and d["fit_utc"].endswith("Z")
    assert {"python", "numpy", "pandas", "scikit-learn", "lightgbm"} <= d["versions"].keys()

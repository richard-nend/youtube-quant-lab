"""publish.sh: 공개용 클론에서만 커밋·push, 데몬이 도는 체크아웃은 pull 하지 않는다(리뷰 I7). 로컬 git 만 쓴다(네트워크 없음)."""
import os
import shutil
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "deploy" / "publish.sh"
ENV = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}


def git(*a, cwd=None):
    return subprocess.run(["git", *a], cwd=cwd, env=ENV, check=True, capture_output=True, text=True).stdout.strip()


def test_publish_from_separate_clone_leaves_running_checkout_alone(tmp_path):
    remote, live, other = tmp_path / "remote.git", tmp_path / "live", tmp_path / "other"
    git("init", "-q", "--bare", "-b", "main", str(remote))
    git("clone", "-q", str(remote), str(live))
    (live / "README").write_text("v1")
    git("add", "README", cwd=live)
    git("commit", "-q", "-m", "v1", cwd=live)
    git("push", "-q", "origin", "HEAD:main", cwd=live)
    git("clone", "-q", str(remote), str(other))                    # 누군가 main 에 새 코드를 올린다
    (other / "README").write_text("v2")
    git("commit", "-qam", "v2", cwd=other)
    git("push", "-q", "origin", "HEAD:main", cwd=other)
    (live / "league" / "deploy").mkdir(parents=True)
    shutil.copy(SCRIPT, live / "league" / "deploy" / "publish.sh")
    pub = live / "league" / "public"
    pub.mkdir()
    for f in ("scoreboard.json", "trades.csv", "reasons.jsonl"):
        (pub / f).write_text(f)
    head = git("rev-parse", "HEAD", cwd=live)
    r = subprocess.run(["/bin/zsh", str(live / "league" / "deploy" / "publish.sh")], capture_output=True, text=True,
                       env={**ENV, "LEAGUE_PUBLISH_DIR": str(tmp_path / "pubclone")})
    assert r.returncode == 0, r.stdout + r.stderr
    assert git("rev-parse", "HEAD", cwd=live) == head and (live / "README").read_text() == "v1"   # 돌던 코드 그대로
    log = git("log", "--format=%s", "main", cwd=remote).splitlines()
    assert log[0].startswith("league: scoreboard") and log[1] == "v2"
    assert git("show", "main:league/public/scoreboard.json", cwd=remote) == "scoreboard.json"

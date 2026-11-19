#!/bin/zsh
# 성적판 파일 3개만 별도 공개용 클론(LEAGUE_PUBLISH_DIR, 기본 ~/league/publish)에서 커밋해 올린다.
# 데몬이 돌고 있는 체크아웃은 읽기만 한다: 거기서 pull 하면 main 의 새 코드가 다음 재시작 때 몰래 돌게 된다(리뷰 I7).
# 코드 배포는 사람이 명시적으로 한다. 원본·판단·상태는 올리지 않는다. 실패 시 3회까지 다시.
set -u
SRC="${0:A:h:h:h}"                                   # 이 스크립트가 있는 체크아웃(성적판을 쓴 곳)
PUB="${LEAGUE_PUBLISH_DIR:-$HOME/league/publish}"
BRANCH="${LEAGUE_PUBLISH_BRANCH:-main}"
files=(league/public/scoreboard.json league/public/trades.csv league/public/reasons.jsonl)
if [ ! -d "$PUB/.git" ]; then
  git clone -q "$(git -C "$SRC" remote get-url origin)" "$PUB" || exit 1
fi
for i in 1 2 3; do
  # 공개용 클론은 매번 원격 그대로로 맞춘 뒤(그 클론에만 있는 변경은 버림) 성적판만 덮어쓴다
  git -C "$PUB" fetch -q origin "$BRANCH" &&
    git -C "$PUB" checkout -q -f -B "$BRANCH" "origin/$BRANCH" &&
    mkdir -p "$PUB/league/public" &&
    for f in $files; do cp "$SRC/$f" "$PUB/$f" || false; done &&
    git -C "$PUB" add -- $files &&
    { git -C "$PUB" diff --cached --quiet -- $files ||
      git -C "$PUB" commit -q -m "league: scoreboard $(date -u +%Y-%m-%dT%H:%MZ)" -- $files; } &&
    git -C "$PUB" push -q origin "HEAD:$BRANCH" && exit 0
  echo "publish attempt $i failed $(date -u)"
  [ $i -lt 3 ] && sleep 30
done
exit 1

#!/bin/zsh
# 성적판 파일만 커밋해서 올린다(데몬이 하루 1회 --publish 로 실행). 원본·판단·상태는 올리지 않는다. 실패 시 3회까지 다시.
set -u
cd "$(dirname "$0")/../.."
files=(league/public/scoreboard.json league/public/trades.csv league/public/reasons.jsonl)
for i in 1 2 3; do
  git add -- $files &&
    { git diff --cached --quiet -- $files || git commit -q -m "league: scoreboard $(date -u +%Y-%m-%dT%H:%MZ)" -- $files; } &&
    git pull -q --rebase --autostash && git push -q && exit 0
  echo "publish attempt $i failed $(date -u)"
  [ $i -lt 3 ] && sleep 30
done
exit 1

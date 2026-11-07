#!/bin/zsh
# Hourly job (launchd, minute 3). Paper only. Commits + pushes when a new signal/trade appears, plus a daily heartbeat.
set -u
export PATH=/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4
cd "$(dirname "$0")/.."
git pull -q --rebase --autostash || echo "pull failed $(date -u)"
uv run --with vendor/laya/laya-0.4.22-py3-none-any.whl python live/runner.py || { echo "runner failed $(date -u)"; exit 1; }
git add live/signals.jsonl live/trades.csv live/status.json 2>/dev/null
if ! git diff --cached --quiet -- live/signals.jsonl live/trades.csv || [ "$(date -u +%H)" = "00" ]; then
  git commit -q -m "live: $(date -u +%Y-%m-%dT%H:%MZ)" && git push -q || echo "push failed $(date -u)"
else
  git reset -q live/status.json
fi

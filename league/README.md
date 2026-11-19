# league

AI 트레이딩 리그(Claude·GPT·Gemini·우리 팀)를 Toobit 모의 체결 엔진으로 돌리는 코드. 규칙은 [RULES.md](RULES.md). 결과는 모두 모의다.

## 실행

`<worktree>`는 이 저장소 체크아웃, `<dir>`은 원본·상태가 쌓이는 데이터 폴더(git에 올리지 않는다).

```
uv run --project <worktree>/league python -m league.daemon --data <dir> [--publish]
uv run --project <worktree>/league python -m league.jobs decide --data <dir>
uv run --project <worktree>/league python -m league.ours.brake
```

- `league.daemon`: 상시 프로세스. 수집(Toobit·바이낸스) -> 원본 저장 -> 엔진 -> 매분 판단 적용·계좌 파일 -> 매시 우리 팀 신호 -> 하루 1회 성적판. `--publish`를 주면 성적판을 쓴 뒤 `deploy/publish.sh`를 실행한다. `--repo`는 성적판을 쓸 저장소 위치(기본: 이 체크아웃).
- `league.jobs decide`: 세 AI 팀(claude, gpt, gemini)을 차례로 한 번씩 판단해 `<dir>/decisions/<팀>/<UTC시각>.json`에 쓴다. 데몬이 다음 분에 가장 새 판단만 적용한다.
- `league.ours.brake`: 우리 팀 브레이크를 EP08b 풀 전체로 적합해 `config/brake.pkl`을 다시 만든다. 시즌 시작에 1회만 하고 이후 동결.

## 환경변수

- `QL_MAIN_ROOT`: 연구 코드·바이낸스 캐시·`runs/ep08b/rows.jsonl`이 있는 quant_lab 본 체크아웃(없으면 개발 맥 경로).
- `QL_EP09_RESULTS`: 브레이크 시험이 읽는 `ep09.json` 경로(시험 전용).
- `VERTEX_PROJECT`: Gemini Vertex 프로젝트(없으면 `config/models.json`의 `vertex_project`).
- `LEAGUE_DATA`: 판단 작업의 기본 데이터 폴더(plist가 `--data`와 같은 값으로 설정).

## 배포 (peri)

`deploy/`의 plist 둘을 `~/Library/LaunchAgents/`에 두고 `launchctl load`한다. 경로는 `/Users/peri/league/...` 기준이라 기계가 다르면 고친다.

- `com.quantlab.league.daemon.plist`: 상시 재시작(KeepAlive).
- `com.quantlab.league.decide.plist`: 4시간마다 판단. launchd는 로컬 시각이라 KST로 적었다(`date +%Z` 확인, KST가 아니면 다시 계산).
- `publish.sh`: 성적판 3개 파일(`public/scoreboard.json`, `trades.csv`, `reasons.jsonl`)만 커밋하고 `pull --rebase` 후 push, 3회까지 재시도.
- 로그인 필요: `claude`, `codex`, `agy`, `gcloud auth application-default login`. 잠자기 끔 확인: `pmset -g`.

## 죽었다 살아났을 때

데이터 폴더 `state/`에서 자동 복구한다. 손으로 할 일은 없고, 재시작만 하면 된다.

- `league.pkl`: 팀별 엔진 스냅샷(판단 적용·우리 팀 진입·종료 때 저장). 시작할 때 읽고, 스냅샷 이후 `raw/`에 쌓인 Toobit 원본을 다시 엔진에 넣는다(마지막 flush 뒤 것은 잃는다).
- `applied.txt`: 이미 적용한 AI 판단 id(`팀:UTC시각`). 먼저 기록하므로 두 번 적용되지 않는다.
- `ours_seen.txt`: 이미 처리한 우리 팀 신호 키. 재시작해도 다시 진입하지 않는다.
- `daemon.json`: 시작 중간가, 마지막 우리 팀 회차, 열린 우리 팀 거래(시간 종료 추적), 판단별 순자산.
- 죽어 있던 시간의 우리 팀 시간별 회차는 따라잡지 않고(경고 로그), 그 시간의 신호는 들어가지 않는다.
- `account_<팀>.json`: 판단 작업이 읽는 계좌 요약(매분 갱신).

## 시험

```
uv run --project <worktree>/league pytest
```

기본은 `-m "not live"`다(`pyproject.toml`). `-m live`는 실제 모델 CLI·API를 부르므로 호출 비용이 든다.

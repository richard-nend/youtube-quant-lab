# 실시간 모의거래 (Live paper test)

**실제 돈을 쓰지 않는 모의거래입니다. 투자 권유가 아닙니다.** Paper trading only — no orders, no keys, not investment advice.

- 규칙: `PREREGISTRATION.md` H-EP07-LIVE, Amendment 3. 시작 시각은 `start.json`.
- 매시 3분(UTC)에 `runner.py`가 바이낸스 닫힌 봉으로 같은 백테스트 엔진을 다시 돌려 새 신호를 기록합니다.
- `signals.jsonl`: 결과를 알기 전에 기록된 신호(Laya 점수 포함). `late=true`는 봉이 닫히고 70분 넘게 지나 기록된 것.
- `trades.csv`: 엔진이 만든 매매(`open=True`는 아직 진행 중). `status.json`: 마지막 실행, 트랙별 모의 잔고(시작 10,000 USDT).
- 트랙: 1 등록 구성(Laya 다국어판 + 위험 2%) · 2 전진 분석만 · 3 Laya 영어판 · 4 위험 0.5%(탐색용).

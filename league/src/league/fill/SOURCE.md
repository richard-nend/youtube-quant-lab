# 출처

AITrading `engine/src/aitrading_engine/*.py` 복사본 (커밋 e4916b47206a85efdd11cb482e2beff69f7b4551, ENGINE_VERSION 0.3.2).
런타임에 AITrading을 import하지 않는다. 변경은 `aitrading_engine` -> `league.fill` import 경로 치환뿐.

- 원본 시험: 58개 통과 (`pytest engine/tests`)
- 복사본 시험: 58개 통과 (`tests/fill`). 시험의 스키마 경로만 `tests/fill/schemas/`(common.json + trading/*.json 복사)로 변경.

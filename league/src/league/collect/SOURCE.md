# 출처

AITrading `collectors/src/aitrading_collectors/` 복사본 (커밋 30c07f2c3efaa8091b3409557a012f1125ed07fa).
런타임에 AITrading을 import하지 않는다.

- 복사: models(Trade·BookSnapshot·MarketCtx·Funding·Candle·Instrument·to_row), util(Throttle·f·now_ms + instrument_ids 일부), venues/base(웹소켓 재연결·폴링), toobit_usdtm(파서·수집기), binance_usdm(parse_agg_trade·parse_depth·웹소켓)
- 뺌: 백필·아카이브·공백 메우기, 바이낸스 mark/forceOrder/OI/펀딩 조회, HL·기타 거래소, sink·config(yaml)
- 변경: SourceConfig -> CollectorSpec(base.py), grade 부여 삭제, import 경로, Toobit 심볼은 BTCUSDT -> BTC-SWAP-USDT 자동 변환

"""Toobit USDT 무기한 규격·수수료. 값은 config/toobit.json (출처·확인일 포함)."""
import json
from decimal import Decimal
from pathlib import Path

from .fill import EngineConfig, FeeSchedule, InstrumentSpec

TOOBIT = "toobit_usdtm"
BTC = "BTC-USDT-PERP"
ETH = "ETH-USDT-PERP"

# shortcut: 저장소 체크아웃(editable 설치) 기준 경로, 휠 배포 시 package-data로 옮길 것
CFG = json.loads((Path(__file__).resolve().parents[2] / "config" / "toobit.json").read_text())


def league_engine_config(initial_cash: Decimal = Decimal(5000)) -> EngineConfig:
    inst = {
        iid: InstrumentSpec(iid, TOOBIT, max_leverage=CFG[k]["max_leverage"], mmr=CFG[k]["mmr"],
                            maint_amount=CFG[k]["maint_amount"], quote="USDT")
        for iid, k in ((BTC, "BTC"), (ETH, "ETH"))
    }
    return EngineConfig(
        instruments=inst, initial_cash=initial_cash,
        fees={TOOBIT: FeeSchedule(CFG["taker"], CFG["maker"])},
        trigger_by={"hyperliquid": "mark", "binance_usdm": "last", TOOBIT: CFG["trigger_by"]},
        liq_clearance_fee={TOOBIT: Decimal(CFG["liq_clearance_fee"])},
    )

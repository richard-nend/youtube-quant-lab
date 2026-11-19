import json
from decimal import Decimal
from pathlib import Path

import pytest

from league.fill import FeeSchedule
from league.venues import BTC, CFG, ETH, TOOBIT, league_engine_config

EXAMPLES_PATH = Path(__file__).parent / "fixtures" / "toobit_liq_examples.json"


def test_config_uses_toobit_values():
    c = league_engine_config()
    assert c.initial_cash == Decimal(5000)
    assert c.fees[TOOBIT] == FeeSchedule(CFG["taker"], CFG["maker"])
    assert c.instruments[BTC].venue == TOOBIT and c.instruments[BTC].quote == "USDT"
    assert c.instruments[BTC].mmr == Decimal(CFG["BTC"]["mmr"])
    assert c.instruments[ETH].mmr == Decimal(CFG["ETH"]["mmr"])
    assert c.trigger_by[TOOBIT] == CFG["trigger_by"]


@pytest.mark.skipif(not EXAMPLES_PATH.exists(), reason="needs Toobit calculator examples")
def test_liquidation_matches_toobit_examples():
    from league.fill.margin import binance_liquidation_price
    for ex in json.loads(EXAMPLES_PATH.read_text()):
        side = 1 if ex["side"] == "long" else -1
        lp = binance_liquidation_price(Decimal(ex["wallet"]), Decimal(ex["qty"]), Decimal(ex["entry"]), side,
                                       Decimal(CFG["BTC"]["mmr"]), Decimal(CFG["BTC"]["maint_amount"]))
        assert abs(lp - Decimal(ex["expected_liq"])) / Decimal(ex["expected_liq"]) < Decimal("0.005")

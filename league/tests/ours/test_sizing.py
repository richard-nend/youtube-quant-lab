from decimal import Decimal as D

from league.ours.sizing import size_qty


def test_size_risk_one_percent():
    assert size_qty(D(5000), D(100), D(98), D(0)) == D(25)


def test_size_capped_by_leverage():
    # 1% risk on a 0.1 stop wants 500 units = 50,000 notional; 3x cap leaves 15,000 - 12,000 open = 3,000 -> 30 units
    assert size_qty(D(5000), D(100), D("99.9"), D(12000)) == D(30)
    assert size_qty(D(5000), D(100), D("99.9"), D(16000)) == D(0)


def test_size_short_and_lot_step():
    assert size_qty(D(5000), D(100), D(103), D(0)) == D("16.666")
    assert size_qty(D(5000), D(100), D(100), D(0)) == D(0)

import json
import logging
from decimal import Decimal
from types import SimpleNamespace

import pandas as pd

from league import daemon as dm
from league.collect.models import BookSnapshot, MarketCtx, Trade
from league.venues import BTC, ETH, TOOBIT

H = 3_600_000
T0 = 1_800_000_000_000 // H * H + 10 * 60_000   # hh:10:00 UTC
SYM = {BTC: "BTC-SWAP-USDT", ETH: "ETH-SWAP-USDT"}


def records(ts):
    out = []
    for iid, s in SYM.items():
        out.append(BookSnapshot(ts, TOOBIT, s, [[99.0, 500.0]], [[100.0, 500.0]], ts))
        out.append(Trade(ts + 1, TOOBIT, s, 100.0, 0.01, "buy", f"t{ts}", ts + 1))
    out.append(MarketCtx(ts + 2, TOOBIT, "BTC-SWAP-USDT", ts + 2, mark_price=99.5))
    out.append(MarketCtx(ts + 2, TOOBIT, "ETH-SWAP-USDT", ts + 2, mark_price=99.5))
    return out


def run(d, t0, secs):
    for i in range(secs):
        ts = t0 + i * 1000
        for r in records(ts):
            d.on_record(r)
        d.tick(ts)


def write_decision(root, team="claude", utc="20270115T0800", pct=100):
    p = root / "decisions" / team / f"{utc}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"team": team, "decided_utc": utc, "status": "ok", "items": [
        {"symbol": "BTC", "action": "long", "notional_pct": pct, "order": "market", "reason": "r"}]}))
    return p


def test_one_minute_applies_decision_once_and_snapshots(tmp_path):
    d = dm.Daemon(tmp_path, repo=tmp_path)
    run(d, T0, 5)                       # 시세가 생긴 뒤에 판단 파일이 나타난다
    write_decision(tmp_path)
    run(d, T0 + 5_000, 70)              # 다음 분 경계를 지난다
    assert (tmp_path / "state" / "applied.txt").read_text().split() == ["claude:20270115T0800"]
    pos = d.lg.view("claude").position(BTC)
    assert pos > 0
    assert (tmp_path / "state" / "league.pkl").exists()
    run(d, T0 + 75_000, 70)             # 또 한 분: 다시 적용하지 않는다
    assert d.lg.view("claude").position(BTC) == pos
    assert all(d.lg.view(t).position(BTC) == 0 for t in ("gpt", "gemini", "ours"))
    assert d.stats["events"] > 0 and d.stats["late"] == 0
    acct = json.loads((tmp_path / "state" / "account_claude.json").read_text())
    assert Decimal(acct["positions"][0]["qty"]) == pos and acct["positions"][0]["symbol"] == "BTC"
    assert {"equity", "cash", "open_orders"} <= acct.keys()
    d.close()
    assert list((tmp_path / "raw").rglob("*.parquet"))   # 멈출 때 버퍼가 디스크로

    d2 = dm.Daemon(tmp_path, repo=tmp_path)             # 재시작: 복원, 중복 적용 없음
    assert d2.lg.view("claude").position(BTC) == pos
    run(d2, T0 + 150_000, 65)
    assert d2.lg.view("claude").position(BTC) == pos


def test_only_latest_unapplied_decision_is_applied_and_result_filled(tmp_path):
    d = dm.Daemon(tmp_path, repo=tmp_path)
    run(d, T0, 5)
    write_decision(tmp_path, utc="20270115T0400", pct=50)
    write_decision(tmp_path, utc="20270115T0800", pct=100)
    run(d, T0 + 5_000, 70)
    assert (tmp_path / "state" / "applied.txt").read_text().split() == ["claude:20270115T0400", "claude:20270115T0800"]
    eq = d.lg.view("claude").equity()
    assert d.lg.view("claude").position(BTC) * 100 / eq * 100 > Decimal(90)   # 명목 % (가격 ≈ 100), 100%짜리, 50%는 건너뜀
    old = json.loads((tmp_path / "decisions/claude/20270115T0400.json").read_text())
    new = json.loads((tmp_path / "decisions/claude/20270115T0800.json").read_text())
    assert old["result"] == {"applied": False}
    assert new["result"]["applied"] and "equity_change_pct" in new["result"]


def test_late_event_is_dropped_and_counted(tmp_path):
    d = dm.Daemon(tmp_path, repo=tmp_path)
    run(d, T0, 10)
    d.on_record(Trade(T0, TOOBIT, "BTC-SWAP-USDT", 100.0, 0.01, "buy", "old", T0))
    d.tick(T0 + 20_000)
    assert d.stats["late"] == 1


def sig(key="BTCUSDT|FVG-1h|x", side=1):
    return SimpleNamespace(key=key, symbol="BTCUSDT", candidate="FVG-1h", signal_time=pd.Timestamp("2027-01-15", tz="UTC"),
                           side=side, entry="market", limit_price=None, stop=98.0, target=106.0, facts={"f": 1.0})


class FakeBrake:
    def __init__(self, block=False):
        self.block = block

    def blocks(self, facts):
        return self.block, 0.5


def test_ours_signal_entered_once_with_stop_and_target(tmp_path):
    d = dm.Daemon(tmp_path, repo=tmp_path, brake=FakeBrake())
    run(d, T0, 5)
    hour = T0 // H * H
    d.run_hour(hour, [sig()])
    d.run_hour(hour, [sig()])           # 같은 key 다시: 무시
    run(d, T0 + 5_000, 5)
    v = d.lg.view("ours")
    assert v.position(BTC) == Decimal("33.333")    # 위험 1% = 50 / (마크 99.5 - 손절 98)
    kinds = sorted(o.req.type for o in v.open_orders(BTC))
    assert kinds == ["stop_market", "take_profit_market"]
    assert (tmp_path / "state" / "league.pkl").exists()
    d.close()
    d2 = dm.Daemon(tmp_path, repo=tmp_path, brake=FakeBrake())
    d2.run_hour(hour + H, [sig()])      # 재시작 후에도 다시 들어가지 않는다
    assert (tmp_path / "state" / "ours_seen.txt").read_text().split() == ["BTCUSDT|FVG-1h|x"]
    assert len(d2.lg.queues["ours"].pending) == 0


def test_brake_block_skips_but_marks_seen(tmp_path):
    d = dm.Daemon(tmp_path, repo=tmp_path, brake=FakeBrake(block=True))
    run(d, T0, 5)
    d.run_hour(T0 // H * H, [sig()])
    run(d, T0 + 5_000, 5)
    assert d.lg.view("ours").position(BTC) == 0
    assert "BTCUSDT|FVG-1h|x" in (tmp_path / "state" / "ours_seen.txt").read_text()


def test_missed_hours_logged(tmp_path, caplog):
    d = dm.Daemon(tmp_path, repo=tmp_path, brake=FakeBrake())
    hour = T0 // H * H
    d.run_hour(hour, [])
    with caplog.at_level(logging.WARNING, logger="league.daemon"):
        d.run_hour(hour + 3 * H, [])
    assert "missed 2" in caplog.text


def test_low_memory_skips_round(tmp_path, monkeypatch, caplog):
    called = []
    d = dm.Daemon(tmp_path, repo=tmp_path, brake=FakeBrake(), signals=lambda now: called.append(now) or [])
    monkeypatch.setattr(dm, "enough_memory", lambda: False)
    with caplog.at_level(logging.WARNING, logger="league.daemon"):
        assert d.fetch_signals(T0 // H * H) is None
    assert not called and "memory" in caplog.text


def test_due_hour_after_delay(tmp_path):
    d = dm.Daemon(tmp_path, repo=tmp_path)
    hour = T0 // H * H
    assert d.due_hour(hour + 30_000) is None        # 정각 + 1분 전에는 아직
    assert d.due_hour(hour + dm.OURS_DELAY_MS) == hour
    d.run_hour(hour, [])
    assert d.due_hour(hour + dm.OURS_DELAY_MS + 1000) is None


def test_daily_scoreboard_written(tmp_path):
    d = dm.Daemon(tmp_path, repo=tmp_path)
    day = T0 // 86_400_000 * 86_400_000
    run(d, day - 3_000, 2)
    run(d, day + 5_000, 3)
    board = json.loads((tmp_path / "league" / "public" / "scoreboard.json").read_text())
    assert board["simulated"] is True and len(board["teams"]) == 4


def test_restart_replays_flushed_records_after_snapshot(tmp_path):
    d = dm.Daemon(tmp_path, repo=tmp_path)
    run(d, T0, 5)
    d.snapshot()
    snap_ts = d.lg.last_ts
    run(d, T0 + 5_000, 70)          # 분 경계에서 원본이 디스크로, 스냅샷은 없음(비정상 종료 흉내)
    flushed_upto = d.lg.last_ts
    d2 = dm.Daemon(tmp_path, repo=tmp_path)
    assert snap_ts < d2.lg.last_ts and d2.lg.last_ts >= flushed_upto - 60_000
    assert d2.stats["events"] > 0


def test_time_exit_values_come_from_candidate_sim():
    assert dm.sim_exits("FVG-1h") == (H, 20, 96)       # ep03 cost_aware_smc.SIM
    assert dm.sim_exits("FLOW-4h") == (4 * H, 10, 12)  # simulate 기본 limit_ttl, CANDIDATES max_hold


def test_unfilled_limit_entry_cancelled_at_limit_ttl(tmp_path):
    hour = T0 // H * H
    s = sig()
    s.entry, s.limit_price, s.signal_time = "limit", 90.0, pd.Timestamp(hour - H, unit="ms", tz="UTC")
    d = dm.Daemon(tmp_path, repo=tmp_path, brake=FakeBrake())
    run(d, T0, 5)
    d.run_hour(hour, [s])
    run(d, T0 + 5_000, 5)
    expire = hour - H + 21 * H                        # FVG-1h limit_ttl 20: 신호 봉 다음 20개 봉 마감까지
    run(d, expire - 120_000, 65)
    assert [o.req.type for o in d.lg.view("ours").open_orders(BTC)] == ["limit"]   # 체결 전: 손절·익절 없음
    run(d, expire + 1_000, 65)
    assert d.lg.view("ours").open_orders(BTC) == [] and d.lg.view("ours").position(BTC) == 0
    assert d.st["ours_open"] == {}


def test_position_closed_at_max_hold(tmp_path):
    hour = T0 // H * H
    s = sig()
    s.signal_time = pd.Timestamp(hour - H, unit="ms", tz="UTC")
    d = dm.Daemon(tmp_path, repo=tmp_path, brake=FakeBrake())
    run(d, T0, 5)
    d.run_hour(hour, [s])
    run(d, T0 + 5_000, 65)
    assert d.lg.view("ours").position(BTC) > 0
    close = hour + 97 * H                             # 체결 봉(hour) + max_hold 96 봉의 마감
    run(d, close - 180_000, 65)
    assert d.lg.view("ours").position(BTC) > 0
    run(d, close + 1_000, 65)
    assert d.lg.view("ours").position(BTC) == 0 and d.lg.view("ours").open_orders(BTC) == []
    assert d.st["ours_open"] == {}


def test_legs_placed_after_limit_fill_and_replaced_if_cancelled(tmp_path):
    hour = T0 // H * H
    s = sig()
    s.entry, s.limit_price, s.signal_time = "limit", 90.0, pd.Timestamp(hour - H, unit="ms", tz="UTC")
    d = dm.Daemon(tmp_path, repo=tmp_path, brake=FakeBrake())
    run(d, T0, 5)
    d.run_hour(hour, [s])
    run(d, T0 + 5_000, 10)
    v = d.lg.view("ours")
    assert [o.req.type for o in v.open_orders(BTC)] == ["limit"] and v.position(BTC) == 0
    t = T0 + 20_000
    for i in range(5):                                # 매도 호가가 지정가 아래로 → 체결
        ts = t + i * 1000
        d.on_record(BookSnapshot(ts, TOOBIT, "BTC-SWAP-USDT", [[88.0, 500.0]], [[89.0, 500.0]], ts))
        d.on_record(Trade(ts + 1, TOOBIT, "BTC-SWAP-USDT", 89.0, 0.01, "sell", f"x{ts}", ts + 1))
        d.tick(ts)
    run(d, t + 5_000, 8)
    assert v.position(BTC) > 0
    legs = {o.req.type: o for o in v.open_orders(BTC)}
    assert set(legs) == {"stop_market", "take_profit_market"} and all(o.req.reduce_only for o in legs.values())
    d.lg.submit("ours", [dm.Cancel(ord_id=legs["stop_market"].ord_id)])   # 손절이 어떤 이유로 사라짐
    run(d, t + 13_000, 8)
    assert {o.req.type for o in v.open_orders(BTC)} == {"stop_market", "take_profit_market"}


def test_failing_job_does_not_stop_tick_and_day_advances(tmp_path, monkeypatch):
    d = dm.Daemon(tmp_path, repo=tmp_path)
    day = T0 // 86_400_000 * 86_400_000
    run(d, day - 3_000, 2)
    (tmp_path / "decisions" / "claude").mkdir(parents=True)
    (tmp_path / "decisions" / "claude" / "20270115T0000.json").write_text("{corrupt")
    boom = lambda *a: (_ for _ in ()).throw(RuntimeError("boom"))
    monkeypatch.setattr(d, "time_exits", boom)
    run(d, day + 5_000, 3)                            # 손상 판단 파일 + 실패하는 작업에도 tick 은 계속
    assert d.st["board_day"] == dm._utc(day, "%Y-%m-%d")
    assert (tmp_path / "league" / "public" / "scoreboard.json").exists()
    monkeypatch.setattr(d, "scoreboard", boom)
    run(d, day + 86_400_000 + 5_000, 3)
    assert d.st["board_day"] == dm._utc(day + 86_400_000, "%Y-%m-%d")   # 성적판이 실패해도 날짜는 넘어간다
    assert (tmp_path / "state" / "account_claude.json").exists()

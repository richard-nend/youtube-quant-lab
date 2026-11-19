from league.jobs import decide_all


def test_decide_all_three_teams_in_order_one_crash_does_not_stop(tmp_path):
    seen = []

    def fake(team, now, root):
        seen.append((team, now, root))
        if team == "claude":
            raise RuntimeError("boom")

    decide_all(tmp_path, 123, decide=fake)
    assert seen == [("claude", 123, tmp_path), ("gpt", 123, tmp_path), ("gemini", 123, tmp_path)]

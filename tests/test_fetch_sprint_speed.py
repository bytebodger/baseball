import pandas as pd

from src.data import fetch_sprint_speed


def _fake_leaderboard(player_ids, sprint_speeds):
    return pd.DataFrame(
        {
            "last_name, first_name": ["Player, Test"] * len(player_ids),
            "player_id": player_ids,
            "team_id": [100] * len(player_ids),
            "team": ["DET"] * len(player_ids),
            "position": ["CF"] * len(player_ids),
            "age": [27] * len(player_ids),
            "competitive_runs": [100] * len(player_ids),
            "bolts": [10] * len(player_ids),
            "hp_to_1b": [4.2] * len(player_ids),
            "sprint_speed": sprint_speeds,
        }
    )


def test_fetch_season_sprint_speed_renames_player_id_and_adds_season(monkeypatch):
    monkeypatch.setattr(
        fetch_sprint_speed, "statcast_sprint_speed", lambda year, min_opp: _fake_leaderboard([123, 456], [28.5, 30.1])
    )

    result = fetch_sprint_speed.fetch_season_sprint_speed(2023)

    assert result["batter_id"].tolist() == [123, 456]
    assert result["season"].tolist() == [2023, 2023]
    assert result["sprint_speed"].tolist() == [28.5, 30.1]
    assert list(result.columns) == ["batter_id", "season", "sprint_speed"]


def test_fetch_and_save_season_skips_when_already_pulled(tmp_path, monkeypatch):
    (tmp_path / "season=2020").mkdir(parents=True)

    def fail_if_called(*args, **kwargs):
        raise AssertionError("statcast_sprint_speed() should not be called when the season is already pulled")

    monkeypatch.setattr(fetch_sprint_speed, "statcast_sprint_speed", fail_if_called)

    fetch_sprint_speed.fetch_and_save_season(2020, output_dir=tmp_path, force=False)


def test_fetch_and_save_season_writes_partitioned_table(tmp_path, monkeypatch):
    monkeypatch.setattr(
        fetch_sprint_speed, "statcast_sprint_speed", lambda year, min_opp: _fake_leaderboard([1, 2], [27.0, 29.5])
    )

    fetch_sprint_speed.fetch_and_save_season(2021, output_dir=tmp_path)

    assert (tmp_path / "season=2021").exists()
    from src.data.statcast_common import read_partitioned

    saved = read_partitioned(tmp_path)
    assert set(saved["batter_id"].tolist()) == {1, 2}
    assert saved["season"].unique().tolist() == [2021]


def test_fetch_and_save_season_handles_empty_leaderboard_without_crashing(tmp_path, monkeypatch):
    monkeypatch.setattr(
        fetch_sprint_speed, "statcast_sprint_speed", lambda year, min_opp: _fake_leaderboard([], [])
    )

    fetch_sprint_speed.fetch_and_save_season(2022, output_dir=tmp_path)

    assert not (tmp_path / "season=2022").exists()


def test_main_pulls_each_year_in_range(tmp_path, monkeypatch):
    calls = []

    def fake(year, min_opp):
        calls.append(year)
        return _fake_leaderboard([1], [28.0])

    monkeypatch.setattr(fetch_sprint_speed, "statcast_sprint_speed", fake)

    fetch_sprint_speed.main(["--start-year", "2015", "--end-year", "2017", "--output-dir", str(tmp_path)])

    assert calls == [2015, 2016, 2017]
    assert (tmp_path / "season=2015").exists()
    assert (tmp_path / "season=2016").exists()
    assert (tmp_path / "season=2017").exists()

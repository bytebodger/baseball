import numpy as np
import pandas as pd
import pytest

from src.data.contact_quality import (
    HARD_HIT_THRESHOLD_MPH,
    ContactQualityHistory,
    babip_features_batch,
    babip_for,
    build_contact_quality_history,
    build_default_histories,
    contact_quality_aux_targets_batch,
    contact_quality_features_batch,
    contact_quality_features_for,
    load_contact_quality_histories,
    load_raw_batted_balls,
    save_contact_quality_histories,
)


def _raw_fixture() -> pd.DataFrame:
    # pitcher 100 allows 3 real batted-ball events (type=="X"): a single
    # (2023-01-01), a home run (2023-02-01, excluded from BABIP), and a
    # field out (2023-03-01). pitcher 200's only row is a FOUL BALL
    # (type=="S") that still carries a real launch_speed -- Statcast tracks
    # exit velocity on fouls too, so this row must be excluded even though
    # launch_speed.notna() would keep it; this is exactly the bug this
    # fixture is here to catch a regression of.
    return pd.DataFrame(
        {
            "pitcher": [100, 100, 100, 200],
            "batter": [900, 900, 901, 902],
            "game_date": ["2023-01-01", "2023-02-01", "2023-03-01", "2023-01-15"],
            "game_pk": [1, 2, 3, 4],
            "type": ["X", "X", "X", "S"],
            "events": ["single", "home_run", "field_out", None],
            "description": ["hit_into_play", "hit_into_play", "hit_into_play", "foul"],
            "launch_speed": [90.0, 100.0, 80.0, 95.0],
            "launch_angle": [10, 20, 5, 60],
            "estimated_ba_using_speedangle": [0.3, 1.0, 0.05, None],
            "at_bat_number": [1, 1, 1, 1],
            "pitch_number": [1, 1, 1, 1],
        }
    )


def test_load_raw_batted_balls_excludes_fouls_even_with_real_launch_speed(tmp_path):
    """Regression test: a foul ball (type=="S") with a real, non-null
    launch_speed must be excluded -- only type=="X" (a real batted-ball
    event) counts. Filtering on launch_speed.notna() alone would wrongly
    keep pitcher 200's foul-ball row here."""
    _raw_fixture().to_parquet(tmp_path / "statcast_2023.parquet")
    batted_balls = load_raw_batted_balls(raw_dir=tmp_path)
    assert len(batted_balls) == 3  # the foul-ball row (pitcher 200) is dropped
    assert set(batted_balls["pitcher_id"]) == {100}
    assert batted_balls["launch_speed"].tolist() == pytest.approx([90.0, 100.0, 80.0])


def test_load_raw_batted_balls_computes_hard_hit_flag_at_the_threshold(tmp_path):
    fixture = _raw_fixture()
    fixture.loc[1, "launch_speed"] = HARD_HIT_THRESHOLD_MPH  # exactly at the boundary
    fixture.to_parquet(tmp_path / "statcast_2023.parquet")
    batted_balls = load_raw_batted_balls(raw_dir=tmp_path)
    hard_hit_by_speed = dict(zip(batted_balls["launch_speed"], batted_balls["hard_hit"]))
    assert hard_hit_by_speed[90.0] == 0.0
    assert hard_hit_by_speed[HARD_HIT_THRESHOLD_MPH] == 1.0  # >= threshold counts as hard-hit
    assert hard_hit_by_speed[80.0] == 0.0


def test_load_raw_batted_balls_season_filtering(tmp_path):
    _raw_fixture().to_parquet(tmp_path / "statcast_2023.parquet")
    only_2023 = load_raw_batted_balls(raw_dir=tmp_path, season_start=2023, season_end=2023)
    assert len(only_2023) == 3  # all real batted-ball rows are 2023
    none_in_2022 = load_raw_batted_balls(raw_dir=tmp_path, season_start=2015, season_end=2022)
    assert len(none_in_2022) == 0


def test_build_default_histories_season_range_override(tmp_path):
    """Walk-forward retraining needs a non-default season boundary --
    confirms season_start/season_end actually change which raw seasons get
    included, not just that they're accepted."""
    _raw_fixture().to_parquet(tmp_path / "statcast_2023.parquet")
    fixture_2015 = _raw_fixture()
    fixture_2015["game_date"] = "2015-01-01"
    fixture_2015["game_pk"] = fixture_2015["game_pk"] + 1000
    fixture_2015.to_parquet(tmp_path / "statcast_2015.parquet")

    only_2015 = build_default_histories(raw_dir=tmp_path, season_start=2015, season_end=2015)
    both_seasons = build_default_histories(raw_dir=tmp_path, season_start=2015, season_end=2023)

    # pitcher 100 has 3 real batted-ball events per season -- restricting to
    # 2015 only should give it half the total events both_seasons sees.
    assert len(only_2015["pitcher"].dates_by_player[100]) == 3
    assert len(both_seasons["pitcher"].dates_by_player[100]) == 6


def test_load_raw_batted_balls_classifies_home_run_and_babip_hit(tmp_path):
    _raw_fixture().to_parquet(tmp_path / "statcast_2023.parquet")
    batted_balls = load_raw_batted_balls(raw_dir=tmp_path)
    by_outcome = batted_balls.set_index("outcome")
    assert by_outcome.loc["single", "is_home_run"] == False  # noqa: E712
    assert by_outcome.loc["single", "is_babip_hit"] == 1.0
    assert by_outcome.loc["home_run", "is_home_run"] == True  # noqa: E712
    assert by_outcome.loc["home_run", "is_babip_hit"] == 0.0  # home runs are never a "BABIP hit"
    # raw events="field_out" maps to the canonical "hit_into_play_out" bucket
    # (see statcast_common.EVENT_OUTCOME_MAP) -- same mapping the processed
    # pitch table's own `outcome` column uses.
    assert by_outcome.loc["hit_into_play_out", "is_home_run"] == False  # noqa: E712
    assert by_outcome.loc["hit_into_play_out", "is_babip_hit"] == 0.0


def _synthetic_batted_balls() -> pd.DataFrame:
    # pitcher 100: single (Jan), home_run (Feb, excluded from BABIP), field_out (Mar).
    # pitcher 200: field_out (mid-Jan).
    return pd.DataFrame(
        {
            "pitcher_id": [100, 100, 100, 200],
            "batter_id": [900, 900, 901, 902],
            "game_date": pd.to_datetime(["2023-03-01", "2023-01-01", "2023-02-01", "2023-01-15"]),
            "launch_speed": [80.0, 90.0, 100.0, 70.0],
            "hard_hit": [0.0, 0.0, 1.0, 0.0],
            "outcome": ["field_out", "single", "home_run", "field_out"],
            "is_home_run": [False, False, True, False],
            "is_babip_hit": [0.0, 1.0, 0.0, 0.0],
        }
    )


def test_build_contact_quality_history_sorts_each_players_events_by_date():
    history = build_contact_quality_history(_synthetic_batted_balls(), "pitcher_id")
    # Pitcher 100's rows were given out of date order (March, Jan, Feb) --
    # the built arrays must be sorted ascending by date, not insertion order.
    dates = history.dates_by_player[100]
    assert list(dates) == sorted(dates)
    assert history.exit_velo_by_player[100].tolist() == pytest.approx([90.0, 100.0, 80.0])  # Jan, Feb, Mar order


def test_build_contact_quality_history_league_average_covers_every_row():
    history = build_contact_quality_history(_synthetic_batted_balls(), "pitcher_id")
    assert history.league_avg_exit_velo == pytest.approx((80.0 + 90.0 + 100.0 + 70.0) / 4)
    assert history.league_avg_hard_hit_rate == pytest.approx((0.0 + 0.0 + 1.0 + 0.0) / 4)


def test_build_contact_quality_history_babip_excludes_home_runs():
    history = build_contact_quality_history(_synthetic_batted_balls(), "pitcher_id")
    # Pitcher 100's babip-relevant history should be just [single (Jan), field_out (Mar)]
    # -- the Feb home run excluded entirely, not counted as a non-hit.
    dates = history.babip_dates_by_player[100]
    assert len(dates) == 2
    assert history.babip_hit_by_player[100].tolist() == pytest.approx([1.0, 0.0])  # Jan single, Mar out
    # League BABIP average: 2 hits (pitcher 100's single, nothing else) / 3 non-HR rows
    # (Jan single, Mar out, pitcher 200's Jan-15 out).
    assert history.league_avg_babip == pytest.approx(1 / 3)


def test_contact_quality_features_for_uses_only_strictly_prior_events():
    history = build_contact_quality_history(_synthetic_batted_balls(), "pitcher_id")
    jan_ns = pd.Timestamp("2023-01-01").value
    feb_ns = pd.Timestamp("2023-02-01").value
    mar_ns = pd.Timestamp("2023-03-01").value

    # Querying exactly at pitcher 100's first event date excludes that same-
    # day event -- "strictly prior," matching every other rolling feature in
    # this project (league_rates_for, PitcherWorkloadHistory, ...). No prior
    # events at all -> falls back to league average.
    avg, rate = contact_quality_features_for(history, 100, jan_ns, min_events=0)
    assert (avg, rate) == (history.league_avg_exit_velo, history.league_avg_hard_hit_rate)

    # As of Feb 1 (exclusive): only the Jan 1 event (90.0, not hard-hit) is prior.
    avg, rate = contact_quality_features_for(history, 100, feb_ns, min_events=0)
    assert avg == pytest.approx(90.0)
    assert rate == pytest.approx(0.0)

    # As of Mar 1 (exclusive): Jan 1 (90.0) and Feb 1 (100.0, hard-hit) are both prior.
    avg, rate = contact_quality_features_for(history, 100, mar_ns, min_events=0)
    assert avg == pytest.approx((90.0 + 100.0) / 2)
    assert rate == pytest.approx(0.5)

    # Well after every real event: all 3 of pitcher 100's events are prior.
    late_ns = pd.Timestamp("2023-06-01").value
    avg, rate = contact_quality_features_for(history, 100, late_ns, min_events=0)
    assert avg == pytest.approx((90.0 + 100.0 + 80.0) / 3)


def test_contact_quality_features_for_falls_back_to_league_average_below_min_events():
    history = build_contact_quality_history(_synthetic_batted_balls(), "pitcher_id")
    late_ns = pd.Timestamp("2023-06-01").value
    # Pitcher 100 has exactly 3 prior events by June; requiring 5 forces the fallback.
    avg, rate = contact_quality_features_for(history, 100, late_ns, min_events=5)
    assert (avg, rate) == (history.league_avg_exit_velo, history.league_avg_hard_hit_rate)
    # Requiring 3 (exactly the count available) should NOT fall back.
    avg, rate = contact_quality_features_for(history, 100, late_ns, min_events=3)
    assert avg == pytest.approx((90.0 + 100.0 + 80.0) / 3)


def test_contact_quality_features_for_falls_back_to_league_average_for_unknown_player():
    history = build_contact_quality_history(_synthetic_batted_balls(), "pitcher_id")
    avg, rate = contact_quality_features_for(history, 999999, pd.Timestamp("2023-06-01").value)
    assert (avg, rate) == (history.league_avg_exit_velo, history.league_avg_hard_hit_rate)


def test_contact_quality_features_batch_matches_single_lookup_row_by_row():
    history = build_contact_quality_history(_synthetic_batted_balls(), "pitcher_id")
    player_ids = pd.Series([100, 100, 200, 999999])
    game_dates = pd.Series(pd.to_datetime(["2023-02-15", "2023-06-01", "2023-06-01", "2023-06-01"]))

    batch_result = contact_quality_features_batch(history, player_ids, game_dates, min_events=0)
    for i, (pid, date) in enumerate(zip(player_ids, game_dates)):
        expected = contact_quality_features_for(history, int(pid), pd.Timestamp(date).value, min_events=0)
        assert batch_result[i].tolist() == pytest.approx(list(expected))


# ---------- babip_for / contact_quality_aux_targets_batch ----------


def test_babip_for_uses_only_strictly_prior_non_home_run_events():
    history = build_contact_quality_history(_synthetic_batted_balls(), "pitcher_id")
    late_ns = pd.Timestamp("2023-06-01").value
    # Pitcher 100's babip-relevant history: single (Jan, hit) + field_out (Mar, not a hit).
    # The Feb home run must NOT appear in this computation at all.
    babip = babip_for(history, 100, late_ns, min_events=0)
    assert babip == pytest.approx(0.5)  # 1 hit / 2 non-HR events


def test_babip_for_falls_back_to_league_average_below_min_events():
    history = build_contact_quality_history(_synthetic_batted_balls(), "pitcher_id")
    late_ns = pd.Timestamp("2023-06-01").value
    babip = babip_for(history, 100, late_ns, min_events=5)
    assert babip == pytest.approx(history.league_avg_babip)


def test_babip_for_falls_back_to_league_average_for_unknown_player():
    history = build_contact_quality_history(_synthetic_batted_balls(), "pitcher_id")
    babip = babip_for(history, 999999, pd.Timestamp("2023-06-01").value)
    assert babip == pytest.approx(history.league_avg_babip)


def test_contact_quality_aux_targets_batch_matches_babip_for_and_hard_hit_rate():
    history = build_contact_quality_history(_synthetic_batted_balls(), "pitcher_id")
    player_ids = pd.Series([100, 200])
    game_dates = pd.Series(pd.to_datetime(["2023-06-01", "2023-06-01"]))

    targets = contact_quality_aux_targets_batch(history, player_ids, game_dates, min_events=0)
    for i, pid in enumerate(player_ids):
        expected_babip = babip_for(history, int(pid), pd.Timestamp("2023-06-01").value, min_events=0)
        _, expected_hard_hit = contact_quality_features_for(history, int(pid), pd.Timestamp("2023-06-01").value, min_events=0)
        assert targets[i, 0] == pytest.approx(expected_babip)
        assert targets[i, 1] == pytest.approx(expected_hard_hit)


def test_babip_features_batch_matches_babip_for_and_has_no_hard_hit_column():
    history = build_contact_quality_history(_synthetic_batted_balls(), "pitcher_id")
    player_ids = pd.Series([100, 200])
    game_dates = pd.Series(pd.to_datetime(["2023-06-01", "2023-06-01"]))

    babips = babip_features_batch(history, player_ids, game_dates, min_events=0)
    assert babips.shape == (2,)
    for i, pid in enumerate(player_ids):
        expected = babip_for(history, int(pid), pd.Timestamp("2023-06-01").value, min_events=0)
        assert babips[i] == pytest.approx(expected)


def test_save_and_load_contact_quality_histories_round_trip(tmp_path):
    pitcher_history = build_contact_quality_history(_synthetic_batted_balls(), "pitcher_id")
    batter_history = build_contact_quality_history(_synthetic_batted_balls(), "batter_id")
    path = tmp_path / "contact_quality.pkl"
    save_contact_quality_histories(pitcher_history, batter_history, path)

    loaded = load_contact_quality_histories(path)
    assert isinstance(loaded["pitcher"], ContactQualityHistory)
    assert isinstance(loaded["batter"], ContactQualityHistory)
    assert loaded["pitcher"].dates_by_player.keys() == pitcher_history.dates_by_player.keys()
    assert loaded["pitcher"].league_avg_exit_velo == pytest.approx(pitcher_history.league_avg_exit_velo)
    assert loaded["pitcher"].league_avg_babip == pytest.approx(pitcher_history.league_avg_babip)

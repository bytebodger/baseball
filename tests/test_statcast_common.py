import pandas as pd

from src.data import statcast_common as sc


def _raw_frame():
    """A tiny fabricated raw Statcast frame covering the outcome-encoding edge cases:
    balls/strikes leading to a walk and a strikeout, a home run, a field error
    (events fallback bucket), and a catcher's interference call (description
    fallback, since events isn't in EVENT_OUTCOME_MAP)."""
    return pd.DataFrame(
        {
            "pitcher": [100, 100, 100, 100, 100, 100, 100],
            "batter": [1, 1, 1, 1, 2, 3, 4],
            "game_date": ["2024-06-15"] * 7,
            "game_pk": [1, 1, 1, 1, 1, 2, 2],
            "at_bat_number": [1, 1, 1, 1, 2, 1, 1],
            "pitch_number": [1, 2, 3, 4, 1, 2, 1],
            "pitch_type": ["FF", "FF", "FF", "FF", "SL", "CH", "SI"],
            "release_speed": [95.0, 95.1, 95.2, 95.3, 88.0, 86.0, 93.0],
            "release_spin_rate": [2200, 2210, 2220, 2230, 2500, None, 2100],
            "spin_rate_deprecated": [None, None, None, None, None, 1900, None],
            "plate_x": [0.1] * 7,
            "plate_z": [2.5] * 7,
            "balls": [0, 1, 2, 3, 0, 1, 0],
            "strikes": [0, 0, 0, 0, 0, 1, 0],
            "outs_when_up": [0, 0, 0, 0, 1, 2, 0],
            "on_1b": [None, None, None, None, 5, None, None],
            "on_2b": [None] * 7,
            "on_3b": [None] * 7,
            "home_score": [0, 0, 0, 0, 0, 1, 1],
            "away_score": [0, 0, 0, 0, 0, 0, 0],
            "n_thruorder_pitcher": [1, 1, 1, 1, 1, 1, 1],
            "inning": [1, 1, 1, 1, 1, 3, 4],
            "inning_topbot": ["Top", "Top", "Top", "Top", "Top", "Bot", "Bot"],
            "stand": ["R", "R", "R", "R", "L", "R", "L"],
            "p_throws": ["L"] * 7,
            "home_team": ["DET"] * 7,
            "away_team": ["CLE"] * 7,
            "game_year": [2024] * 7,
            "events": [None, None, None, "walk", "home_run", "field_error", "catcher_interf"],
            "description": [
                "ball",
                "ball",
                "ball",
                "ball",
                "hit_into_play",
                "hit_into_play",
                "called_strike",
            ],
        }
    )


def test_compute_outcome_priorities_events_over_description():
    events = pd.Series(["strikeout", None, "field_error", "catcher_interf"])
    description = pd.Series(["swinging_strike", "foul", "hit_into_play", "called_strike"])
    outcome = sc.compute_outcome(events, description)
    assert outcome.tolist() == ["strikeout", "foul", "hit_into_play_out", "called_strike"]


def test_compute_outcome_unmapped_is_null():
    events = pd.Series(["not_a_real_event"])
    description = pd.Series(["also_not_real"])
    outcome = sc.compute_outcome(events, description)
    assert outcome.isna().all()


def test_best_spin_rate_coalesces():
    raw = _raw_frame()
    spin = sc.best_spin_rate(raw)
    assert spin.tolist() == [2200, 2210, 2220, 2230, 2500, 1900, 2100]


def test_flag_missing_critical():
    df = pd.DataFrame({"a": [1, None, 3], "b": [None, None, 6]})
    is_valid, missing_fields = sc.flag_missing_critical(df, ["a", "b"])
    assert is_valid.tolist() == [False, False, True]
    assert missing_fields.tolist() == ["b", "a,b", ""]


def test_build_pitch_frame_from_raw_columns_and_outcomes():
    df = sc.build_pitch_frame_from_raw(_raw_frame())
    assert list(df["pitcher_id"]) == [100] * 7
    assert list(df["batter_id"]) == [1, 1, 1, 1, 2, 3, 4]
    assert df["outcome"].tolist() == [
        "ball",
        "ball",
        "ball",
        "walk",
        "home_run",
        "hit_into_play_out",
        "called_strike",
    ]
    assert df["spin_rate"].tolist() == [2200, 2210, 2220, 2230, 2500, 1900, 2100]


def test_build_pitch_frame_from_raw_adds_game_state_and_park_columns():
    df = sc.build_pitch_frame_from_raw(_raw_frame())
    assert df["on_1b"].tolist()[4] == 5
    assert df["on_1b"].isna().tolist() == [True, True, True, True, False, True, True]
    assert df["on_2b"].isna().all()
    assert df["home_score"].tolist() == [0, 0, 0, 0, 0, 1, 1]
    assert df["away_score"].tolist() == [0] * 7
    # n_thruorder_pitcher is 1 (first encounter) everywhere in this fixture,
    # so times_through_order (prior encounters before this PA) is 0 everywhere.
    assert df["times_through_order"].tolist() == [0] * 7
    # DET never changed parks in configs/park_history.yaml, so park_id defaults to the team code.
    assert df["park_id"].tolist() == ["DET"] * 7


def test_write_partitioned_creates_season_directory(tmp_path):
    df = sc.build_pitch_frame_from_raw(_raw_frame())
    df["is_valid"] = True
    output_dir = tmp_path / "pitches"
    sc.write_partitioned(df, output_dir)

    assert (output_dir / "season=2024").exists()


def test_read_partitioned_round_trips_nullable_season_column(tmp_path):
    # game_year (source of `season`) is a pandas nullable Int64 column when read
    # from real Statcast parquet files -- pyarrow then dictionary-encodes the
    # `season` partition column on write, which plain pd.read_parquet(dir) can't
    # decode back (NotImplementedError on dictionary<values=int32>). Use the
    # nullable dtype here so this test actually exercises that path.
    raw = _raw_frame()
    raw["game_year"] = pd.array(raw["game_year"], dtype="Int64")
    df = sc.build_pitch_frame_from_raw(raw)
    df["is_valid"] = True

    output_dir = tmp_path / "pitches"
    sc.write_partitioned(df, output_dir)

    written = sc.read_partitioned(output_dir)
    assert len(written) == len(df)
    assert set(written["season"].unique().tolist()) == {2024}

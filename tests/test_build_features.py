import pandas as pd

from src.data import statcast_common as sc
from src.data.build_features import CRITICAL_FIELDS, build_season_pitches_from_frame


def _pitch_frame():
    """Two pitchers, out of chronological order and out of pitcher order, so the
    sort in build_season_pitches_from_frame is actually exercised. Pitcher 200's
    lone pitch is missing release_speed, so it should come back flagged."""
    raw = pd.DataFrame(
        {
            "pitcher": [100, 200, 100],
            "batter": [1, 2, 1],
            "game_date": ["2024-06-16", "2024-06-15", "2024-06-15"],
            "game_pk": [2, 3, 1],
            "at_bat_number": [1, 1, 1],
            "pitch_number": [1, 1, 1],
            "pitch_type": ["FF", "SL", "CH"],
            "release_speed": [95.0, None, 86.0],
            "release_spin_rate": [2200, 2400, 2100],
            "spin_rate_deprecated": [None, None, None],
            "plate_x": [0.1, 0.2, 0.3],
            "plate_z": [2.5, 2.4, 2.3],
            "balls": [0, 0, 0],
            "strikes": [0, 0, 0],
            "outs_when_up": [0, 1, 0],
            "on_1b": [None, None, None],
            "on_2b": [None, None, None],
            "on_3b": [None, None, None],
            "home_score": [0, 0, 0],
            "away_score": [0, 0, 0],
            "n_thruorder_pitcher": [1, 1, 1],
            "inning": [1, 2, 1],
            "inning_topbot": ["Top", "Bot", "Top"],
            "stand": ["R", "L", "R"],
            "p_throws": ["L", "R", "L"],
            "home_team": ["DET", "CLE", "DET"],
            "away_team": ["CLE", "DET", "CLE"],
            "game_year": [2024, 2024, 2024],
            "events": ["single", "strikeout", "walk"],
            "description": ["hit_into_play", "swinging_strike", "ball"],
        }
    )
    return sc.build_pitch_frame_from_raw(raw)


def test_sorted_by_pitcher_then_chronologically():
    result = build_season_pitches_from_frame(_pitch_frame())
    # pitcher 100's two pitches (2024-06-15 then 2024-06-16) come before pitcher 200's.
    assert result["pitcher_id"].tolist() == [100, 100, 200]
    assert result["game_date"].dt.strftime("%Y-%m-%d").tolist() == [
        "2024-06-15",
        "2024-06-16",
        "2024-06-15",
    ]


def test_missing_release_speed_is_flagged_not_dropped():
    result = build_season_pitches_from_frame(_pitch_frame())
    assert len(result) == 3  # nothing dropped

    flagged = result[result["pitcher_id"] == 200].iloc[0]
    assert not flagged["is_valid"]
    assert "release_speed" in flagged["missing_fields"]

    clean_rows = result[result["pitcher_id"] == 100]
    assert clean_rows["is_valid"].all()


def test_critical_fields_cover_all_requested_columns():
    expected = {
        "pitcher_id",
        "batter_id",
        "game_date",
        "game_pk",
        "at_bat_number",
        "pitch_number",
        "pitch_type",
        "release_speed",
        "spin_rate",
        "plate_x",
        "plate_z",
        "balls",
        "strikes",
        "outs_when_up",
        "home_score",
        "away_score",
        "times_through_order",
        "inning",
        "stand",
        "p_throws",
        "home_team",
        "away_team",
        "outcome",
    }
    assert set(CRITICAL_FIELDS) == expected

import pandas as pd

from src.data import statcast_common as sc
from src.data.build_plate_appearances import build_season_plate_appearances_from_frame


def _pitch_frame():
    """One 4-pitch walk (batter 1) and one 1-pitch home run (batter 2), plus a
    second game for batter 1 the day before, to check the batter-then-chronological
    sort."""
    raw = pd.DataFrame(
        {
            "pitcher": [100, 100, 100, 100, 100, 100],
            "batter": [1, 1, 1, 1, 2, 1],
            "game_date": ["2024-06-16"] * 4 + ["2024-06-16", "2024-06-15"],
            "game_pk": [2, 2, 2, 2, 2, 1],
            "at_bat_number": [1, 1, 1, 1, 2, 1],
            "pitch_number": [1, 2, 3, 4, 1, 1],
            "pitch_type": ["FF", "FF", "FF", "FF", "SL", "CH"],
            "release_speed": [95.0, 95.1, 95.2, 95.3, 88.0, 86.0],
            "release_spin_rate": [2200, 2210, 2220, 2230, 2500, 2100],
            "spin_rate_deprecated": [None] * 6,
            "plate_x": [0.1] * 6,
            "plate_z": [2.5] * 6,
            "balls": [0, 1, 2, 3, 0, 0],
            "strikes": [0, 0, 0, 0, 0, 0],
            "outs_when_up": [0, 0, 0, 0, 1, 2],
            "on_1b": [None] * 6,
            "on_2b": [None] * 6,
            "on_3b": [None] * 6,
            "home_score": [0] * 6,
            "away_score": [0] * 6,
            "n_thruorder_pitcher": [1] * 6,
            "inning": [1, 1, 1, 1, 1, 3],
            "inning_topbot": ["Top"] * 6,
            "stand": ["R", "R", "R", "R", "L", "R"],
            "p_throws": ["L"] * 6,
            "home_team": ["DET"] * 6,
            "away_team": ["CLE"] * 6,
            "game_year": [2024] * 6,
            "events": [None, None, None, "walk", "home_run", "single"],
            "description": ["ball", "ball", "ball", "ball", "hit_into_play", "hit_into_play"],
        }
    )
    return sc.build_pitch_frame_from_raw(raw)


def test_one_row_per_plate_appearance():
    result = build_season_plate_appearances_from_frame(_pitch_frame())
    assert len(result) == 3  # 2 PAs in game_pk=2, 1 PA in game_pk=1


def test_walk_outcome_and_final_count_and_pitch_count():
    result = build_season_plate_appearances_from_frame(_pitch_frame())
    walk_row = result[(result["game_pk"] == 2) & (result["at_bat_number"] == 1)].iloc[0]

    assert walk_row["outcome"] == "walk"
    assert walk_row["pitch_count"] == 4
    # count entering the final (4th) pitch, not the resolved post-pitch count
    assert walk_row["balls"] == 3
    assert walk_row["strikes"] == 0


def test_sorted_by_batter_then_chronologically():
    result = build_season_plate_appearances_from_frame(_pitch_frame())
    batter_1_rows = result[result["batter_id"] == 1]
    assert batter_1_rows["game_date"].dt.strftime("%Y-%m-%d").tolist() == [
        "2024-06-15",
        "2024-06-16",
    ]
    assert result["batter_id"].tolist()[-1] == 2  # batter 2 sorts after batter 1

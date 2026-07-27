import pandas as pd
import pytest

from src.evaluation.season_backtest_fixtures import build_fixture, pitcher_and_batter_query_pairs


def _game_row(game_pk=1, home_score=4, away_score=2):
    return pd.Series(
        {
            "game_pk": game_pk,
            "game_date": pd.Timestamp("2025-06-01"),
            "home_team": "DET",
            "away_team": "CLE",
            "home_starter_id": 100,
            "away_starter_id": 200,
            "home_score": home_score,
            "away_score": away_score,
            "home_win": home_score > away_score,
        }
    )


def _appearances(game_pk=1):
    batter_appearances = pd.DataFrame(
        [
            {"game_pk": game_pk, "team": "DET", "batter_id": 10, "first_at_bat_number": 1},
            {"game_pk": game_pk, "team": "DET", "batter_id": 11, "first_at_bat_number": 2},
            {"game_pk": game_pk, "team": "CLE", "batter_id": 20, "first_at_bat_number": 1},
        ]
    )
    pitcher_appearances = pd.DataFrame(
        [
            {"team": "DET", "pitcher_id": 100, "game_date": pd.Timestamp("2025-05-25")},
            {"team": "DET", "pitcher_id": 101, "game_date": pd.Timestamp("2025-05-28")},
            {"team": "CLE", "pitcher_id": 200, "game_date": pd.Timestamp("2025-05-20")},
            {"team": "CLE", "pitcher_id": 201, "game_date": pd.Timestamp("2025-05-30")},
            # Outside the bullpen window (>14 days before 2025-06-01) -- must be excluded.
            {"team": "DET", "pitcher_id": 999, "game_date": pd.Timestamp("2025-04-01")},
        ]
    )
    return batter_appearances, pitcher_appearances


def test_build_fixture_derives_lineup_bullpen_and_actual_outcome_correctly():
    row = _game_row()
    batter_appearances, pitcher_appearances = _appearances()

    fixture = build_fixture(row, batter_appearances, pitcher_appearances)

    assert fixture["home_lineup"] == [10, 11]
    assert fixture["away_lineup"] == [20]
    # Bullpen excludes the starter itself and anything outside the trailing window.
    assert fixture["home_bullpen"] == [101]
    assert fixture["away_bullpen"] == [201]
    assert fixture["actual_home_win"] is True
    assert fixture["actual_home_score"] == 4
    assert fixture["actual_away_score"] == 2


def test_pitcher_and_batter_query_pairs_deduplicates_across_fixtures():
    row1 = _game_row(game_pk=1)
    row2 = _game_row(game_pk=2)  # same date/players -- pairs should collapse, not double up.
    batter_appearances, pitcher_appearances = _appearances(game_pk=1)
    batter_appearances2, pitcher_appearances2 = _appearances(game_pk=2)
    fixtures = [
        build_fixture(row1, batter_appearances, pitcher_appearances),
        build_fixture(row2, batter_appearances2, pitcher_appearances2),
    ]

    pitcher_pairs, batter_pairs = pitcher_and_batter_query_pairs(fixtures)

    date_ns = pd.Timestamp("2025-06-01").value
    assert set(pitcher_pairs) == {(100, date_ns), (101, date_ns), (200, date_ns), (201, date_ns)}
    assert set(batter_pairs) == {(10, date_ns), (11, date_ns), (20, date_ns)}

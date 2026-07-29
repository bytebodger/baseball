import json

import pandas as pd
import pytest

from src.evaluation import season_backtest_progress_synopsis as sbps


def test_synopsis_computes_running_stats_against_real_games_for_completed_subset(tmp_path, monkeypatch):
    results_path = tmp_path / "results.jsonl"
    rows = [
        {"game_pk": 1, "game_date": "2025-04-01", "home_team": "DET", "away_team": "CLE",
         "actual_home_win": True, "model_home_prob": 0.7, "sim_mean_total_runs": 6.0},
        {"game_pk": 2, "game_date": "2025-04-02", "home_team": "DET", "away_team": "CLE",
         "actual_home_win": False, "model_home_prob": 0.4, "sim_mean_total_runs": 8.0},
        {"game_pk": 3, "game_date": "2025-04-03", "home_team": "DET", "away_team": "CLE",
         "actual_home_win": True, "model_home_prob": 0.3, "sim_mean_total_runs": 12.0},  # wrong-side prediction
    ]
    with open(results_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    real_games = pd.DataFrame(
        [
            {"game_pk": 1, "home_score": 4, "away_score": 2, "home_win": True},
            {"game_pk": 2, "home_score": 1, "away_score": 5, "home_win": False},
            {"game_pk": 3, "home_score": 6, "away_score": 3, "home_win": True},
        ]
    )
    monkeypatch.setattr(sbps, "select_season_games_with_betting_lines", lambda season: real_games)

    result = sbps.synopsis(results_path, season=2025)

    assert result["n_games"] == 3
    assert result["mean_sim_total_runs"] == pytest.approx((6.0 + 8.0 + 12.0) / 3)
    assert result["mean_real_total_runs_same_games"] == pytest.approx((6 + 6 + 9) / 3)
    # Predictions: game1 p=0.7>0.5 -> home, actual home_win=True -> correct.
    # game2 p=0.4<0.5 -> away, actual home_win=False -> correct.
    # game3 p=0.3<0.5 -> away, actual home_win=True -> WRONG.
    assert result["win_accuracy"] == pytest.approx(2 / 3)
    expected_brier = ((0.7 - 1) ** 2 + (0.4 - 0) ** 2 + (0.3 - 1) ** 2) / 3
    assert result["brier_score"] == pytest.approx(expected_brier)
    assert result["tier_counts"] == {"<7": 1, "7-9": 1, ">9": 1}


def test_synopsis_ignores_a_truncated_trailing_line_from_a_mid_write_kill(tmp_path, monkeypatch):
    results_path = tmp_path / "results.jsonl"
    good_row = {"game_pk": 1, "game_date": "2025-04-01", "home_team": "DET", "away_team": "CLE",
                "actual_home_win": True, "model_home_prob": 0.6, "sim_mean_total_runs": 9.0}
    with open(results_path, "w") as f:
        f.write(json.dumps(good_row) + "\n")
        f.write('{"game_pk": 2, "model_home_pr')  # truncated, no trailing newline -- simulates a kill mid-write

    real_games = pd.DataFrame([{"game_pk": 1, "home_score": 5, "away_score": 4, "home_win": True}])
    monkeypatch.setattr(sbps, "select_season_games_with_betting_lines", lambda season: real_games)

    result = sbps.synopsis(results_path, season=2025)
    assert result["n_games"] == 1  # the truncated row is skipped, not a crash


def test_synopsis_with_no_results_file_reports_zero_games(tmp_path):
    result = sbps.synopsis(tmp_path / "nonexistent.jsonl", season=2025)
    assert result == {"n_games": 0}

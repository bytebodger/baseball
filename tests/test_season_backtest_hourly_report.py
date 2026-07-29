import json

import pandas as pd
import pytest

from src.evaluation import season_backtest_hourly_report as sbhr
from src.resumable_job import write_progress


def _write_results(path, rows):
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def test_hourly_report_computes_progress_accuracy_runs_and_clv(tmp_path, monkeypatch):
    results_path = tmp_path / "results.jsonl"
    progress_path = tmp_path / "progress.json"

    # game 1: pick 'em open (0.5/0.5), model says 0.60 home -> edge 0.10 > 2% threshold -> bet home.
    #         close shortens home to -150/away +130 -> market moved toward home -> positive CLV. Home wins.
    # game 2: pick 'em open/close (no market movement), model says 0.505 -> edge within threshold -> no bet.
    _write_results(
        results_path,
        [
            {"game_pk": 1, "model_home_prob": 0.60, "sim_mean_total_runs": 8.0},
            {"game_pk": 2, "model_home_prob": 0.505, "sim_mean_total_runs": 10.0},
        ],
    )
    write_progress(progress_path, total=5, completed=2, extra={"season": 2025})

    real_games = pd.DataFrame(
        [
            {"game_pk": 1, "home_win": True, "home_ml_open": -110, "away_ml_open": -110,
             "home_ml_close": -150, "away_ml_close": 130},
            {"game_pk": 2, "home_win": False, "home_ml_open": -110, "away_ml_open": -110,
             "home_ml_close": -110, "away_ml_close": -110},
        ]
    )
    monkeypatch.setattr(sbhr, "select_season_games_with_betting_lines", lambda season: real_games)

    result = sbhr.hourly_report(results_path, progress_path, season=2025)

    assert result["n_completed"] == 2
    assert result["n_total"] == 5
    assert result["n_remaining"] == 3
    assert result["n_joined_with_lines"] == 2
    assert result["mean_sim_total_runs"] == pytest.approx((8.0 + 10.0) / 2)
    # game1: p=0.60>0.5 -> home, actual home_win=True -> correct. game2: p=0.505>0.5 -> home, actual home_win=False -> wrong.
    assert result["win_accuracy"] == pytest.approx(0.5)
    assert result["n_bets_placed"] == 1  # only game 1 clears the edge threshold
    assert result["mean_clv"] > 0  # market moved toward the bet side by close


def test_hourly_report_with_no_completed_games_reports_zero(tmp_path):
    progress_path = tmp_path / "progress.json"
    write_progress(progress_path, total=5, completed=0, extra={})
    result = sbhr.hourly_report(tmp_path / "nonexistent.jsonl", progress_path, season=2025)
    assert result["n_completed"] == 0
    assert result["n_total"] == 5

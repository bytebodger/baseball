from pathlib import Path
from unittest.mock import patch

import pytest

from src.evaluation.season_backtest_simulation import main, parse_args


def test_parse_args_xbh_calibration_defaults_to_off():
    args = parse_args(
        [
            "--season", "2025", "--event-model-checkpoint", "ckpt.pt", "--embedding-cache-dir", "cache",
            "--contact-quality-checkpoint", "cq.pkl", "--event-model-season-start", "2015",
            "--event-model-season-end", "2024", "--results-path", "r.jsonl", "--progress-path", "p.json",
        ]
    )
    assert args.xbh_calibration_gain == 0.0
    assert args.xbh_rate_checkpoint is None


def test_parse_args_xbh_calibration_flags_parse():
    args = parse_args(
        [
            "--season", "2025", "--event-model-checkpoint", "ckpt.pt", "--embedding-cache-dir", "cache",
            "--contact-quality-checkpoint", "cq.pkl", "--event-model-season-start", "2015",
            "--event-model-season-end", "2024", "--results-path", "r.jsonl", "--progress-path", "p.json",
            "--xbh-rate-checkpoint", "xbh.pkl", "--xbh-calibration-gain", "1.0",
        ]
    )
    assert args.xbh_calibration_gain == 1.0
    assert args.xbh_rate_checkpoint == Path("xbh.pkl")


def test_main_raises_when_gain_nonzero_without_checkpoint(tmp_path):
    """Must fail before any data loading -- select_season_games_with_betting_lines
    is deliberately left unmocked/unpatched here so a bug that moved this check
    later would surface as a real, unrelated crash instead of silently passing."""
    argv = [
        "--season", "2025", "--event-model-checkpoint", "ckpt.pt", "--embedding-cache-dir", "cache",
        "--contact-quality-checkpoint", "cq.pkl", "--event-model-season-start", "2015",
        "--event-model-season-end", "2024",
        "--results-path", str(tmp_path / "r.jsonl"), "--progress-path", str(tmp_path / "p.json"),
        "--xbh-calibration-gain", "1.0",
    ]
    with pytest.raises(SystemExit):
        main(argv)


def test_main_passes_xbh_calibration_kwargs_to_build_game_engine_context(tmp_path):
    import pandas as pd

    games = pd.DataFrame({"game_pk": []})
    argv = [
        "--season", "2025", "--event-model-checkpoint", "ckpt.pt", "--embedding-cache-dir", "cache",
        "--contact-quality-checkpoint", "cq.pkl", "--event-model-season-start", "2015",
        "--event-model-season-end", "2024",
        "--results-path", str(tmp_path / "r.jsonl"), "--progress-path", str(tmp_path / "p.json"),
        "--xbh-rate-checkpoint", "xbh.pkl", "--xbh-calibration-gain", "1.0",
    ]
    with patch("src.evaluation.season_backtest_simulation.select_season_games_with_betting_lines", return_value=games), \
         patch("src.evaluation.season_backtest_simulation.load_appearance_tables", return_value=(None, None)), \
         patch("src.evaluation.season_backtest_simulation.build_game_engine_context") as mock_build:
        main(argv)

    assert mock_build.call_count == 1
    _, kwargs = mock_build.call_args
    assert kwargs["xbh_calibration_gain"] == 1.0
    assert kwargs["xbh_rate_checkpoint"] == Path("xbh.pkl")


def test_main_omits_xbh_rate_checkpoint_kwarg_when_not_given(tmp_path):
    import pandas as pd

    games = pd.DataFrame({"game_pk": []})
    argv = [
        "--season", "2025", "--event-model-checkpoint", "ckpt.pt", "--embedding-cache-dir", "cache",
        "--contact-quality-checkpoint", "cq.pkl", "--event-model-season-start", "2015",
        "--event-model-season-end", "2024",
        "--results-path", str(tmp_path / "r.jsonl"), "--progress-path", str(tmp_path / "p.json"),
    ]
    with patch("src.evaluation.season_backtest_simulation.select_season_games_with_betting_lines", return_value=games), \
         patch("src.evaluation.season_backtest_simulation.load_appearance_tables", return_value=(None, None)), \
         patch("src.evaluation.season_backtest_simulation.build_game_engine_context") as mock_build:
        main(argv)

    _, kwargs = mock_build.call_args
    assert kwargs["xbh_calibration_gain"] == 0.0
    assert "xbh_rate_checkpoint" not in kwargs  # let build_game_engine_context's own default apply

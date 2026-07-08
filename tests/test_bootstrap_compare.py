from dataclasses import asdict

import numpy as np
import pandas as pd
import pytest
import torch

from src.data.build_features import build_season_pitches_from_frame
from src.data.sequence_dataset import PlayerPitchSequenceDataset
from src.data.statcast_common import (
    TEST_SEASON_RANGE,
    TRAIN_SEASON_RANGE,
    VAL_SEASONS,
    build_pitch_frame_from_raw,
    write_partitioned,
)
from src.inference.bootstrap_compare import (
    bootstrap_compare,
    main as bootstrap_main,
    summarize_bootstrap,
)
from src.models.game_predictor import GamePredictor, GamePredictorConfig
from src.models.player_encoder import PlayerEncoder, PlayerEncoderConfig
from src.models.set_pooling import PlayerSetPooler, PlayerSetPoolerConfig
from src.training.train_game_predictor import CONTEXT_DIM


def test_bootstrap_compare_shape_and_diff_columns():
    rng = np.random.default_rng(0)
    y_true = (rng.random(200) > 0.5).astype(float)
    probs_a = rng.random(200)
    probs_b = rng.random(200)

    results = bootstrap_compare(y_true, probs_a, probs_b, n_resamples=50, seed=1)

    assert len(results) == 50
    assert set(results.columns) == {
        "accuracy_a", "accuracy_b", "accuracy_diff", "brier_a", "brier_b", "brier_diff",
    }
    assert np.allclose(results["accuracy_diff"], results["accuracy_a"] - results["accuracy_b"])
    assert np.allclose(results["brier_diff"], results["brier_a"] - results["brier_b"])


def test_bootstrap_compare_is_reproducible_with_a_seed():
    rng = np.random.default_rng(0)
    y_true = (rng.random(100) > 0.5).astype(float)
    probs_a = rng.random(100)
    probs_b = rng.random(100)

    first = bootstrap_compare(y_true, probs_a, probs_b, n_resamples=30, seed=42)
    second = bootstrap_compare(y_true, probs_a, probs_b, n_resamples=30, seed=42)

    pd.testing.assert_frame_equal(first, second)


def test_bootstrap_compare_favors_the_clearly_better_method():
    """Method A predicts perfectly; method B predicts the exact opposite --
    every single resample should favor A on both metrics, unanimously."""
    n = 300
    y_true = np.tile([0.0, 1.0], n // 2)
    probs_a = y_true.copy()  # perfect predictions
    probs_b = 1.0 - y_true  # perfectly wrong

    results = bootstrap_compare(y_true, probs_a, probs_b, n_resamples=200, seed=7)
    summary = summarize_bootstrap(results)

    assert summary["fraction_favoring_a_accuracy"] == 1.0
    assert summary["fraction_favoring_b_accuracy"] == 0.0
    assert summary["fraction_favoring_a_brier"] == 1.0
    assert summary["fraction_favoring_b_brier"] == 0.0
    assert summary["accuracy_diff_mean"] == pytest.approx(1.0)
    assert summary["brier_diff_mean"] == pytest.approx(-1.0)


def test_summarize_bootstrap_is_symmetric_for_identical_methods():
    """Two identical prediction arrays: every resample is an exact tie, so
    neither method is ever favored and the mean difference is exactly zero."""
    rng = np.random.default_rng(3)
    y_true = (rng.random(150) > 0.5).astype(float)
    probs = rng.random(150)

    results = bootstrap_compare(y_true, probs, probs, n_resamples=100, seed=5)
    summary = summarize_bootstrap(results)

    assert summary["accuracy_diff_mean"] == 0.0
    assert summary["brier_diff_mean"] == 0.0
    assert summary["fraction_favoring_a_accuracy"] == 0.0
    assert summary["fraction_favoring_b_accuracy"] == 0.0


def test_summarize_bootstrap_ci95_bounds_are_ordered():
    rng = np.random.default_rng(11)
    y_true = (rng.random(400) > 0.5).astype(float)
    probs_a = rng.random(400)
    probs_b = rng.random(400)

    results = bootstrap_compare(y_true, probs_a, probs_b, n_resamples=500, seed=2)
    summary = summarize_bootstrap(results)

    lo, hi = summary["accuracy_diff_ci95"]
    assert lo <= hi
    lo, hi = summary["brier_diff_ci95"]
    assert lo <= hi


def _raw_row(pitcher, batter, game_date, at_bat_number, pitch_number, inning_topbot, home_team, away_team, home_score, away_score, season, game_pk=None):
    return {
        "pitcher": pitcher,
        "batter": batter,
        "game_date": game_date,
        "game_pk": season if game_pk is None else game_pk,
        "game_year": season,
        "game_type": "R",
        "home_team": home_team,
        "away_team": away_team,
        "inning_topbot": inning_topbot,
        "inning": 1,
        "at_bat_number": at_bat_number,
        "pitch_number": pitch_number,
        "pitch_type": "FF",
        "release_speed": 90.0,
        "release_spin_rate": 2200,
        "spin_rate_deprecated": None,
        "plate_x": 0.1,
        "plate_z": 2.2,
        "balls": 0,
        "strikes": 0,
        "outs_when_up": 0,
        "stand": "R",
        "p_throws": "L",
        "events": "field_out",
        "description": "hit_into_play",
        "post_home_score": home_score,
        "post_away_score": away_score,
    }


def _write_fixture(raw_dir, pitches_dir):
    """Games spanning train+val+test (2015-2025): one per season, except the
    val season (2023), which gets a second game with the OPPOSITE outcome so
    it has both classes present -- backtest.py's stacking ensemble fits a
    LogisticRegression on the val season, which needs at least 2 classes."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []
    seasons = list(range(TRAIN_SEASON_RANGE[0], TEST_SEASON_RANGE[1] + 1))
    for season in seasons:
        default_outcome = (5, 3) if season % 2 == 0 else (3, 5)
        outcomes = [default_outcome]
        if season in VAL_SEASONS:
            outcomes.append(default_outcome[::-1])

        season_rows = []
        for game_index, (home_score, away_score) in enumerate(outcomes):
            date = f"{season}-04-{game_index + 1:02d}"
            game_pk = season * 10 + game_index
            season_rows += [
                _raw_row(100, 101 + i, date, i + 1, 1, "Top", "DET", "CLE", home_score, away_score, season, game_pk)
                for i in range(9)
            ] + [
                _raw_row(200, 1 + i, date, 10 + i, 1, "Bot", "DET", "CLE", home_score, away_score, season, game_pk)
                for i in range(9)
            ]

        pd.DataFrame(season_rows).to_parquet(raw_dir / f"statcast_{season}.parquet")
        all_rows.extend(season_rows)

    raw_all = pd.DataFrame(all_rows)
    pitches = build_season_pitches_from_frame(build_pitch_frame_from_raw(raw_all))
    write_partitioned(pitches, pitches_dir)
    return pitches


def _write_game_predictor_checkpoint(path, pitches):
    train_pitches = pitches[pitches["season"].between(*TRAIN_SEASON_RANGE) & pitches["is_valid"]]
    continuous_stats = PlayerPitchSequenceDataset._compute_continuous_stats(train_pitches)

    encoder_config = PlayerEncoderConfig(hidden_size=8, num_layers=1, num_heads=2, dropout=0.0, feedforward_dim=16, max_seq_len=5)
    encoder = PlayerEncoder(encoder_config)

    predictor_config = GamePredictorConfig(
        context_dim=CONTEXT_DIM, hidden_dim=16, num_layers=1, dropout=0.0, runs_distribution="negative_binomial"
    )
    game_predictor = GamePredictor(encoder, predictor_config)

    pooler_config = PlayerSetPoolerConfig(embed_dim=8, num_heads=2, dropout=0.0)
    bullpen_pooler = PlayerSetPooler(pooler_config)
    lineup_pooler = PlayerSetPooler(pooler_config)

    torch.save(
        {
            "game_predictor_state_dict": game_predictor.state_dict(),
            "bullpen_pooler_state_dict": bullpen_pooler.state_dict(),
            "lineup_pooler_state_dict": lineup_pooler.state_dict(),
            "encoder_config": asdict(encoder_config),
            "predictor_config": asdict(predictor_config),
            "pooler_config": asdict(pooler_config),
            "continuous_stats": continuous_stats,
            "rest_day_stats": (4.0, 2.0),
            "epoch": 1,
            "val_loss": 0.1,
        },
        path,
    )


def test_main_runs_end_to_end_and_writes_resamples_and_plot(tmp_path):
    raw_dir = tmp_path / "raw"
    pitches_dir = tmp_path / "pitches"
    pitches = _write_fixture(raw_dir, pitches_dir)

    checkpoint_path = tmp_path / "game_predictor_best.pt"
    _write_game_predictor_checkpoint(checkpoint_path, pitches)

    output_dir = tmp_path / "bootstrap"

    bootstrap_main(
        [
            "--game-predictor-checkpoint", str(checkpoint_path),
            "--pitches-dir", str(pitches_dir),
            "--raw-dir", str(raw_dir),
            "--games-dir", str(tmp_path / "games"),
            "--pitcher-appearances-dir", str(tmp_path / "pitcher_appearances"),
            "--batter-appearances-dir", str(tmp_path / "batter_appearances"),
            "--batch-size", "4",
            "--device", "cpu",
            "--cache-dir", str(tmp_path / "sequence_cache"),
            "--output-dir", str(output_dir),
            "--n-resamples", "25",
            "--seed", "0",
        ]
    )

    results_path = output_dir / "bootstrap_resamples.csv"
    assert results_path.exists()
    results = pd.read_csv(results_path)
    assert len(results) == 25

    plot_path = output_dir / "bootstrap_distributions.png"
    assert plot_path.exists()
    assert plot_path.stat().st_size > 0


def test_main_rejects_unknown_method_names(tmp_path):
    raw_dir = tmp_path / "raw"
    pitches_dir = tmp_path / "pitches"
    pitches = _write_fixture(raw_dir, pitches_dir)

    checkpoint_path = tmp_path / "game_predictor_best.pt"
    _write_game_predictor_checkpoint(checkpoint_path, pitches)

    with pytest.raises(ValueError):
        bootstrap_main(
            [
                "--game-predictor-checkpoint", str(checkpoint_path),
                "--pitches-dir", str(pitches_dir),
                "--raw-dir", str(raw_dir),
                "--games-dir", str(tmp_path / "games"),
                "--pitcher-appearances-dir", str(tmp_path / "pitcher_appearances"),
                "--batter-appearances-dir", str(tmp_path / "batter_appearances"),
                "--batch-size", "4",
                "--device", "cpu",
                "--cache-dir", str(tmp_path / "sequence_cache"),
                "--output-dir", str(tmp_path / "bootstrap"),
                "--method-a", "Not A Real Method",
            ]
        )

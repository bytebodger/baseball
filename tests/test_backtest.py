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
from src.inference.backtest import (
    ALWAYS_HOME_METHOD,
    AVERAGE_ENSEMBLE_METHOD,
    ENSEMBLE_METHOD,
    FEATURE_COLUMNS,
    GAME_PREDICTOR_METHOD,
    LOGISTIC_REGRESSION_METHOD,
    EnsembleArtifacts,
    apply_stacking_ensemble,
    average_ensemble,
    build_baseline_features,
    fit_logistic_regression_baseline,
    fit_stacking_ensemble,
    load_ensemble_artifacts,
    main as backtest_main,
    plot_calibration,
    save_ensemble_artifacts,
    summarize,
)
from src.models.game_predictor import GamePredictor, GamePredictorConfig
from src.models.player_encoder import PlayerEncoder, PlayerEncoderConfig
from src.models.set_pooling import PlayerSetPooler, PlayerSetPoolerConfig


def _fake_games(n=6):
    """A tiny, hand-built games table (not derived from raw pitches) purely
    to exercise the baseline-feature math directly: pitcher 100 starts at
    home every time, allowing progressively more/fewer runs; team DET is
    home every time, scoring a fixed 4 runs; team CLE is away, scoring 2."""
    dates = pd.date_range("2023-04-01", periods=n, freq="D")
    return pd.DataFrame(
        {
            "game_pk": range(1, n + 1),
            "game_date": dates,
            "season": [2023] * n,
            "home_team": ["DET"] * n,
            "away_team": ["CLE"] * n,
            "home_score": [4] * n,
            "away_score": [2, 3, 1, 4, 2, 0][:n],
            "home_win": [True] * n,
            "home_starter_id": [100] * n,
            "away_starter_id": [200] * n,
        }
    )


def test_starter_era_proxy_is_expanding_and_excludes_current_game():
    games = _fake_games()
    result, _ = build_baseline_features(games)

    # First start ever for pitcher 100 has no prior history -> filled with the league average.
    league_avg = float(pd.concat([games["home_score"], games["away_score"]]).mean())
    assert result.loc[0, "home_era_proxy"] == pytest.approx(league_avg)

    # By the 3rd game, pitcher 100's era_proxy should be the average of games 1-2's
    # runs allowed (away_score, since he's the home starter): (2 + 3) / 2 = 2.5.
    assert result.loc[2, "home_era_proxy"] == pytest.approx(2.5)


def test_team_wrc_plus_proxy_indexes_to_100_at_league_average():
    games = _fake_games()
    result, league_avg_runs = build_baseline_features(games)

    # DET's own runs scored (home_score) is a constant 4 every game, so once
    # it has any history its wRC+ proxy should be a fixed 400/league_avg_runs.
    expected = 100.0 * 4.0 / league_avg_runs
    assert result.loc[2, "home_wrc_plus_proxy"] == pytest.approx(expected)


def test_baseline_features_have_no_nulls_after_filling():
    games = _fake_games()
    result, _ = build_baseline_features(games)
    assert not result[FEATURE_COLUMNS].isna().any().any()


def test_average_ensemble_is_the_elementwise_mean():
    a = np.array([1.0, 0.0, 0.4])
    b = np.array([0.5, 0.5, 0.6])

    ensemble = average_ensemble(a, b)

    assert np.allclose(ensemble, [0.75, 0.25, 0.5])


def test_average_ensemble_supports_more_than_two_arrays():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    c = np.array([0.5, 0.5])

    ensemble = average_ensemble(a, b, c)

    assert np.allclose(ensemble, [0.5, 0.5])


def test_fit_stacking_ensemble_learns_to_ignore_an_uninformative_input():
    """gp_probs perfectly predicts home_win; lr_probs is pure noise (constant
    0.5, uninformative). A fitted stacking model should end up relying on
    gp_probs and producing near-perfect predictions -- it must not just
    average the two the way average_ensemble would (which, mixed with a
    constant 0.5, would blur gp_probs' perfect signal)."""
    rng = np.random.default_rng(0)
    home_win = (rng.random(200) > 0.5).astype(float)
    gp_probs = np.clip(home_win * 0.9 + 0.05, 0.01, 0.99)  # strongly informative, not literally 0/1
    lr_probs = np.full(200, 0.5)  # uninformative

    model = fit_stacking_ensemble(gp_probs, lr_probs, home_win)
    stacked = apply_stacking_ensemble(model, gp_probs, lr_probs)

    accuracy = ((stacked >= 0.5) == (home_win == 1.0)).mean()
    assert accuracy > 0.95

    naive_average = average_ensemble(gp_probs, lr_probs)
    naive_accuracy = ((naive_average >= 0.5) == (home_win == 1.0)).mean()
    assert accuracy >= naive_accuracy  # stacking should not do worse than blind averaging here


def test_apply_stacking_ensemble_returns_valid_probabilities():
    rng = np.random.default_rng(1)
    home_win = (rng.random(100) > 0.5).astype(float)
    gp_probs = rng.random(100)
    lr_probs = rng.random(100)

    model = fit_stacking_ensemble(gp_probs, lr_probs, home_win)
    stacked = apply_stacking_ensemble(model, gp_probs, lr_probs)

    assert stacked.shape == (100,)
    assert np.all((stacked >= 0.0) & (stacked <= 1.0))


def test_ensemble_artifacts_round_trip_through_disk(tmp_path):
    rng = np.random.default_rng(2)
    home_win = (rng.random(100) > 0.5).astype(float)
    gp_probs = rng.random(100)
    lr_probs = rng.random(100)

    lr_baseline = fit_logistic_regression_baseline(
        pd.DataFrame({**{col: rng.random(100) for col in FEATURE_COLUMNS}, "home_win": home_win})
    )
    stacking_model = fit_stacking_ensemble(gp_probs, lr_probs, home_win)
    artifacts = EnsembleArtifacts(lr_baseline=lr_baseline, stacking_model=stacking_model)

    path = tmp_path / "ensemble_models.pkl"
    save_ensemble_artifacts(artifacts, path)
    assert path.exists()

    loaded = load_ensemble_artifacts(path)

    # predictions from the reloaded models must match the originals exactly
    original_stacked = apply_stacking_ensemble(artifacts.stacking_model, gp_probs, lr_probs)
    reloaded_stacked = apply_stacking_ensemble(loaded.stacking_model, gp_probs, lr_probs)
    assert np.array_equal(original_stacked, reloaded_stacked)

    era_features = rng.random((10, len(FEATURE_COLUMNS)))
    original_lr = artifacts.lr_baseline.predict_proba(era_features)
    reloaded_lr = loaded.lr_baseline.predict_proba(era_features)
    assert np.array_equal(original_lr, reloaded_lr)


def test_summarize_computes_accuracy_and_brier():
    y_true = np.array([1.0, 0.0, 1.0, 1.0])
    win_prob = np.array([1.0, 1.0, 1.0, 1.0])  # "always predict home win"

    summary = summarize({"always_home": (y_true, win_prob)})

    assert summary.loc[0, "n_games"] == 4
    assert summary.loc[0, "accuracy"] == pytest.approx(0.75)  # wrong on the single away win
    assert summary.loc[0, "brier_score"] == pytest.approx(((1 - y_true) ** 2).mean())


def test_plot_calibration_writes_a_file_including_a_degenerate_constant_baseline(tmp_path):
    y_true = np.array([1.0, 0.0, 1.0, 0.0, 1.0])
    results = {
        "varied": (y_true, np.array([0.9, 0.2, 0.8, 0.3, 0.6])),
        "always_home": (y_true, np.ones(5)),  # constant prediction -- must not crash calibration_curve
    }
    output_path = tmp_path / "calibration.png"

    plot_calibration(results, output_path)

    assert output_path.exists()
    assert output_path.stat().st_size > 0


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
    """Games spanning train+val+test (2015-2025): one game per season, home
    team wins in even seasons/away team wins in odd seasons (so the held-out
    2024 (loss)/2025 (win) games give the "always predict home win" baseline
    a non-degenerate 50% accuracy to check against) -- except the val season
    (2023), which gets a second game with the OPPOSITE outcome, so it has
    both classes present (backtest.py's stacking ensemble fits a
    LogisticRegression on the val season, which needs at least 2 classes)."""
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
        context_dim=6, hidden_dim=16, num_layers=1, dropout=0.0, runs_distribution="negative_binomial"
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


def test_main_runs_end_to_end_and_writes_summary_and_plot(tmp_path):
    raw_dir = tmp_path / "raw"
    pitches_dir = tmp_path / "pitches"
    pitches = _write_fixture(raw_dir, pitches_dir)

    checkpoint_path = tmp_path / "game_predictor_best.pt"
    _write_game_predictor_checkpoint(checkpoint_path, pitches)

    output_dir = tmp_path / "reports"

    backtest_main(
        [
            "--game-predictor-checkpoint", str(checkpoint_path),
            "--pitches-dir", str(pitches_dir),
            "--raw-dir", str(raw_dir),
            "--games-dir", str(tmp_path / "games"),
            "--pitcher-appearances-dir", str(tmp_path / "pitcher_appearances"),
            "--batter-appearances-dir", str(tmp_path / "batter_appearances"),
            "--batch-size", "4",
            "--device", "cpu",
            "--output-dir", str(output_dir),
            "--cache-dir", str(tmp_path / "sequence_cache"),
            "--ensemble-path", str(tmp_path / "ensemble_models.pkl"),
        ]
    )

    summary_path = output_dir / "backtest_summary.csv"
    assert summary_path.exists()
    summary = pd.read_csv(summary_path)
    assert set(summary["method"]) == {
        GAME_PREDICTOR_METHOD, LOGISTIC_REGRESSION_METHOD, ENSEMBLE_METHOD, AVERAGE_ENSEMBLE_METHOD, ALWAYS_HOME_METHOD,
    }
    assert (summary["n_games"] == len(range(TEST_SEASON_RANGE[0], TEST_SEASON_RANGE[1] + 1))).all()

    ensemble_path = tmp_path / "ensemble_models.pkl"
    assert ensemble_path.exists()

    cache_dir = tmp_path / "sequence_cache"
    assert (cache_dir / "pitcher").exists()
    assert any((cache_dir / "pitcher").iterdir())

    always_home_row = summary[summary["method"] == ALWAYS_HOME_METHOD].iloc[0]
    assert always_home_row["accuracy"] == pytest.approx(0.5)  # 1 win (2024), 1 loss (2025) in the fixture
    assert always_home_row["brier_score"] == pytest.approx(0.5)

    plot_path = output_dir / "calibration_plot.png"
    assert plot_path.exists()
    assert plot_path.stat().st_size > 0

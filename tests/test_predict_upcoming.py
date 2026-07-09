from dataclasses import asdict

import numpy as np
import pandas as pd
import pytest
import torch

from src.data.build_features import build_season_pitches_from_frame
from src.data.game_dataset import ensure_game_tables_built
from src.data.sequence_dataset import PlayerPitchSequenceDataset
from src.data.statcast_common import (
    TEST_SEASON_RANGE,
    TRAIN_SEASON_RANGE,
    VAL_SEASONS,
    build_pitch_frame_from_raw,
    read_partitioned,
    write_partitioned,
)
from src.inference.backtest import (
    FEATURE_COLUMNS,
    EnsembleArtifacts,
    build_baseline_features,
    fit_logistic_regression_baseline,
    fit_stacking_ensemble,
    save_ensemble_artifacts,
)
from src.inference.predict_upcoming import (
    DEFAULT_UPCOMING_SEASON,
    main as predict_main,
    parse_args as predict_parse_args,
    predict_upcoming,
)
from src.models.game_predictor import GamePredictor, GamePredictorConfig
from src.models.player_encoder import PlayerEncoder, PlayerEncoderConfig
from src.models.set_pooling import PlayerSetPooler, PlayerSetPoolerConfig
from src.training.train_game_predictor import CONTEXT_DIM


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
        "on_1b": None,
        "on_2b": None,
        "on_3b": None,
        "home_score": 0,
        "away_score": 0,
        "n_thruorder_pitcher": 1,
        "stand": "R",
        "p_throws": "L",
        "events": "field_out",
        "description": "hit_into_play",
        "post_home_score": home_score,
        "post_away_score": away_score,
    }


def _write_fixture(raw_dir, pitches_dir):
    """Games spanning train+val+test+upcoming (2015-2026): one game per
    season, except the val season (2023), which gets a second game with the
    OPPOSITE outcome so it has both classes present (needed to fit the
    stacking LogisticRegression). Season 2026 (DEFAULT_UPCOMING_SEASON)
    stands in for "upcoming" games -- genuinely untouched by
    training/validation/backtesting, same as in the real pipeline."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []
    seasons = list(range(TRAIN_SEASON_RANGE[0], DEFAULT_UPCOMING_SEASON + 1))
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


def _write_ensemble_artifacts(path, raw_dir, games_dir, pitcher_appearances_dir, batter_appearances_dir):
    """Fits real LR baseline + stacking ensemble weights on the fixture's
    own train/val games -- exactly what backtest.py's main() would produce
    (using a stand-in constant for GamePredictor's val-game predictions,
    since fitting the stacking model doesn't require a real model run to
    test predict_upcoming.py's loading/application logic)."""
    all_seasons = list(range(TRAIN_SEASON_RANGE[0], DEFAULT_UPCOMING_SEASON + 1))
    ensure_game_tables_built(all_seasons, raw_dir, games_dir, pitcher_appearances_dir, batter_appearances_dir)
    all_games = read_partitioned(games_dir).sort_values("game_date").reset_index(drop=True)
    all_games, _ = build_baseline_features(all_games)

    train_games = all_games[all_games["season"].between(TRAIN_SEASON_RANGE[0], VAL_SEASONS[-1])]
    lr_baseline = fit_logistic_regression_baseline(train_games)

    val_games = all_games[all_games["season"].isin(VAL_SEASONS)]
    lr_val_probs = lr_baseline.predict_proba(val_games[FEATURE_COLUMNS].to_numpy())[:, 1]
    gp_val_probs = np.full(len(val_games), 0.5)
    stacking_model = fit_stacking_ensemble(gp_val_probs, lr_val_probs, val_games["home_win"].to_numpy())

    save_ensemble_artifacts(EnsembleArtifacts(lr_baseline=lr_baseline, stacking_model=stacking_model), path)


def _common_args(tmp_path, checkpoint_path, pitches_dir, raw_dir, ensemble_path):
    return [
        "--game-predictor-checkpoint", str(checkpoint_path),
        "--pitches-dir", str(pitches_dir),
        "--raw-dir", str(raw_dir),
        "--games-dir", str(tmp_path / "games"),
        "--pitcher-appearances-dir", str(tmp_path / "pitcher_appearances"),
        "--batter-appearances-dir", str(tmp_path / "batter_appearances"),
        "--batch-size", "4",
        "--device", "cpu",
        "--cache-dir", str(tmp_path / "sequence_cache"),
        "--ensemble-path", str(ensemble_path),
    ]


def test_predict_upcoming_prints_a_win_probability_per_game(tmp_path, capsys):
    raw_dir = tmp_path / "raw"
    pitches_dir = tmp_path / "pitches"
    pitches = _write_fixture(raw_dir, pitches_dir)

    checkpoint_path = tmp_path / "game_predictor_best.pt"
    _write_game_predictor_checkpoint(checkpoint_path, pitches)

    ensemble_path = tmp_path / "ensemble_models.pkl"
    _write_ensemble_artifacts(
        ensemble_path, raw_dir, tmp_path / "games", tmp_path / "pitcher_appearances", tmp_path / "batter_appearances"
    )

    output_csv = tmp_path / "predictions.csv"
    predict_main(_common_args(tmp_path, checkpoint_path, pitches_dir, raw_dir, ensemble_path) + ["--output-csv", str(output_csv)])

    assert output_csv.exists()
    predictions = pd.read_csv(output_csv)

    assert len(predictions) == 1  # one game in season 2026 in the fixture
    assert "ensemble_win_prob" in predictions.columns
    assert predictions["ensemble_win_prob"].notna().all()
    assert predictions["ensemble_win_prob"].between(0.0, 1.0).all()
    for col in ["gamepredictor_win_prob", "lr_baseline_win_prob", "home_win_prob", "away_win_prob"]:
        assert predictions[col].between(0.0, 1.0).all()
    # home + away probabilities from the same ensemble must sum to 1
    assert np.allclose(predictions["home_win_prob"] + predictions["away_win_prob"], 1.0)


def test_predict_upcoming_as_of_date_filters_out_earlier_games(tmp_path):
    raw_dir = tmp_path / "raw"
    pitches_dir = tmp_path / "pitches"
    pitches = _write_fixture(raw_dir, pitches_dir)

    checkpoint_path = tmp_path / "game_predictor_best.pt"
    _write_game_predictor_checkpoint(checkpoint_path, pitches)

    ensemble_path = tmp_path / "ensemble_models.pkl"
    _write_ensemble_artifacts(
        ensemble_path, raw_dir, tmp_path / "games", tmp_path / "pitcher_appearances", tmp_path / "batter_appearances"
    )

    args = _common_args(tmp_path, checkpoint_path, pitches_dir, raw_dir, ensemble_path)
    parsed = predict_parse_args(args + ["--as-of-date", f"{DEFAULT_UPCOMING_SEASON}-12-31"])
    predictions = predict_upcoming(parsed)

    assert len(predictions) == 0  # the fixture's only 2026 game is dated April, before Dec 31


def test_predict_upcoming_uses_the_loaded_ensemble_not_a_refit(tmp_path):
    """Two different, deliberately-mismatched ensemble artifact files should
    produce different ensemble_win_prob outputs for the same games -- proof
    predict_upcoming.py is actually applying what it loaded, not silently
    refitting its own stacking model from scratch."""
    raw_dir = tmp_path / "raw"
    pitches_dir = tmp_path / "pitches"
    pitches = _write_fixture(raw_dir, pitches_dir)

    checkpoint_path = tmp_path / "game_predictor_best.pt"
    _write_game_predictor_checkpoint(checkpoint_path, pitches)

    games_dir = tmp_path / "games"
    pitcher_dir = tmp_path / "pitcher_appearances"
    batter_dir = tmp_path / "batter_appearances"

    ensemble_path_a = tmp_path / "ensemble_a.pkl"
    _write_ensemble_artifacts(ensemble_path_a, raw_dir, games_dir, pitcher_dir, batter_dir)

    # a second artifacts file whose stacking model is forced to ignore both
    # inputs and always predict a fixed high probability
    all_seasons = list(range(TRAIN_SEASON_RANGE[0], DEFAULT_UPCOMING_SEASON + 1))
    ensure_game_tables_built(all_seasons, raw_dir, games_dir, pitcher_dir, batter_dir)
    all_games = read_partitioned(games_dir).sort_values("game_date").reset_index(drop=True)
    all_games, _ = build_baseline_features(all_games)
    train_games = all_games[all_games["season"].between(TRAIN_SEASON_RANGE[0], VAL_SEASONS[-1])]
    lr_baseline = fit_logistic_regression_baseline(train_games)

    val_games = all_games[all_games["season"].isin(VAL_SEASONS)]
    always_low = np.zeros(len(val_games))
    stacking_model_b = fit_stacking_ensemble(always_low, always_low, val_games["home_win"].to_numpy())
    ensemble_path_b = tmp_path / "ensemble_b.pkl"
    save_ensemble_artifacts(EnsembleArtifacts(lr_baseline=lr_baseline, stacking_model=stacking_model_b), ensemble_path_b)

    args_a = _common_args(tmp_path, checkpoint_path, pitches_dir, raw_dir, ensemble_path_a)
    args_b = _common_args(tmp_path, checkpoint_path, pitches_dir, raw_dir, ensemble_path_b)

    predictions_a = predict_upcoming(predict_parse_args(args_a))
    predictions_b = predict_upcoming(predict_parse_args(args_b))

    assert not np.allclose(predictions_a["ensemble_win_prob"], predictions_b["ensemble_win_prob"])

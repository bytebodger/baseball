from dataclasses import asdict

import numpy as np
import pandas as pd
import pytest
import torch

from src.data.build_features import build_season_pitches_from_frame
from src.data.sequence_dataset import PlayerPitchSequenceDataset
from src.data.statcast_common import build_pitch_frame_from_raw, write_partitioned
from src.inference.walk_forward_backtest import (
    GP_VS_LR_COMPARISON,
    GP_VS_VEGAS_COMPARISON,
    Fold,
    aggregate_fold_bootstraps,
    generate_folds,
    main as walk_forward_main,
    plot_fold_effects,
)
from src.models.player_encoder import PlayerEncoder, PlayerEncoderConfig


def test_generate_folds_slides_forward_one_year_at_a_time():
    folds = generate_folds(season_range=(2015, 2018), train_years=1, val_years=1, test_years=1)

    assert folds == [
        Fold(0, (2015, 2015), (2016, 2016), (2017, 2017)),
        Fold(1, (2016, 2016), (2017, 2017), (2018, 2018)),
    ]


def test_generate_folds_matches_the_default_5_1_1_window():
    folds = generate_folds(season_range=(2015, 2026), train_years=5, val_years=1, test_years=1)

    assert len(folds) == 6
    assert folds[0].train_seasons == (2015, 2019)
    assert folds[0].val_seasons == (2020, 2020)
    assert folds[0].test_seasons == (2021, 2021)
    assert folds[-1].train_seasons == (2020, 2024)
    assert folds[-1].val_seasons == (2025, 2025)
    assert folds[-1].test_seasons == (2026, 2026)


def test_generate_folds_produces_no_folds_when_range_is_too_short():
    folds = generate_folds(season_range=(2015, 2016), train_years=5, val_years=1, test_years=1)
    assert folds == []


def test_generate_folds_windows_never_overlap_the_next_folds_train_start():
    """Each fold's test season should be exactly the year after its val
    season, and val exactly the year after train -- no gaps, no overlaps
    within a single fold."""
    for fold in generate_folds(season_range=(2015, 2026), train_years=5, val_years=1, test_years=1):
        assert fold.val_seasons[0] == fold.train_seasons[1] + 1
        assert fold.test_seasons[0] == fold.val_seasons[1] + 1


def _bootstrap_summary(fold_index, test_seasons, accuracy_diff_ci95, brier_diff_ci95, comparison=GP_VS_LR_COMPARISON):
    return {
        "fold": fold_index,
        "test_seasons": test_seasons,
        "comparison": comparison,
        "n_resamples": 100,
        "accuracy_diff_mean": sum(accuracy_diff_ci95) / 2,
        "accuracy_diff_std": 0.01,
        "accuracy_diff_ci95": accuracy_diff_ci95,
        "fraction_favoring_a_accuracy": 0.9,
        "fraction_favoring_b_accuracy": 0.1,
        "brier_diff_mean": sum(brier_diff_ci95) / 2,
        "brier_diff_std": 0.001,
        "brier_diff_ci95": brier_diff_ci95,
        "fraction_favoring_a_brier": 0.9,
        "fraction_favoring_b_brier": 0.1,
    }


def test_aggregate_fold_bootstraps_flags_significant_folds_correctly():
    summaries = [
        # CI entirely above zero -> significant win for GamePredictor on accuracy
        _bootstrap_summary(0, "2021-2021", (0.01, 0.05), (-0.01, 0.01)),
        # CI entirely below zero -> significant win for the baseline on accuracy
        _bootstrap_summary(1, "2022-2022", (-0.05, -0.01), (-0.01, 0.01)),
        # CI straddles zero -> inconclusive on accuracy
        _bootstrap_summary(2, "2023-2023", (-0.02, 0.02), (-0.01, 0.01)),
    ]

    aggregate = aggregate_fold_bootstraps(summaries)

    assert aggregate.loc[0, "accuracy_significant_for_gamepredictor"]
    assert not aggregate.loc[0, "accuracy_significant_for_baseline"]

    assert aggregate.loc[1, "accuracy_significant_for_baseline"]
    assert not aggregate.loc[1, "accuracy_significant_for_gamepredictor"]

    assert not aggregate.loc[2, "accuracy_significant_for_gamepredictor"]
    assert not aggregate.loc[2, "accuracy_significant_for_baseline"]


def test_aggregate_fold_bootstraps_brier_significance_uses_lower_is_better():
    # Brier CI entirely below zero -> GamePredictor (lower Brier) favored
    summaries = [_bootstrap_summary(0, "2021-2021", (-0.01, 0.01), (-0.05, -0.01))]
    aggregate = aggregate_fold_bootstraps(summaries)
    assert aggregate.loc[0, "brier_significant_for_gamepredictor"]
    assert not aggregate.loc[0, "brier_significant_for_baseline"]


def test_plot_fold_effects_writes_a_file(tmp_path):
    summaries = [
        _bootstrap_summary(0, "2021-2021", (0.01, 0.05), (-0.01, 0.01)),
        _bootstrap_summary(1, "2022-2022", (-0.02, 0.02), (-0.02, 0.02)),
    ]
    aggregate = aggregate_fold_bootstraps(summaries)
    output_path = tmp_path / "fold_effects.png"

    plot_fold_effects(aggregate, GP_VS_LR_COMPARISON, output_path)

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def _raw_row(pitcher, batter, game_date, at_bat_number, pitch_number, inning_topbot, home_team, away_team, home_score, away_score, season):
    return {
        "pitcher": pitcher,
        "batter": batter,
        "game_date": game_date,
        "game_pk": season,
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


def _write_fixture(raw_dir, pitches_dir, seasons):
    """One game per season, home team wins in even seasons/away team wins in
    odd seasons -- so any 2 consecutive seasons (every fold's train+val
    combined, used to fit the LR baseline) have both classes present."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for season in seasons:
        date = f"{season}-04-01"
        home_score, away_score = (5, 3) if season % 2 == 0 else (3, 5)
        rows = [
            _raw_row(100, 101 + i, date, i + 1, 1, "Top", "DET", "CLE", home_score, away_score, season) for i in range(9)
        ] + [
            _raw_row(200, 1 + i, date, 10 + i, 1, "Bot", "DET", "CLE", home_score, away_score, season) for i in range(9)
        ]
        pd.DataFrame(rows).to_parquet(raw_dir / f"statcast_{season}.parquet")
        all_rows.extend(rows)

    raw_all = pd.DataFrame(all_rows)
    pitches = build_season_pitches_from_frame(build_pitch_frame_from_raw(raw_all))
    write_partitioned(pitches, pitches_dir)
    return pitches


def _write_betting_lines_fixture(betting_lines_dir, game_pks):
    """One betting line per game_pk (matching _write_fixture's one-game-per-season
    convention, where game_pk == season), so every fold's test game has a
    matched Vegas closing line."""
    df = pd.DataFrame(
        {
            "game_pk": game_pks,
            "season": game_pks,
            "home_ml_close": [-150] * len(game_pks),
            "away_ml_close": [130] * len(game_pks),
        }
    )
    df.to_parquet(betting_lines_dir, partition_cols=["season"], index=False)


def _write_encoder_checkpoint(path, pitches):
    continuous_stats = PlayerPitchSequenceDataset._compute_continuous_stats(pitches[pitches["is_valid"]])
    config = PlayerEncoderConfig(hidden_size=8, num_layers=1, num_heads=2, dropout=0.0, feedforward_dim=16, max_seq_len=5)
    encoder = PlayerEncoder(config)
    torch.save(
        {
            "encoder_state_dict": encoder.state_dict(),
            "config": asdict(config),
            "continuous_stats": continuous_stats,
            "epoch": 1,
            "val_loss": 0.1,
        },
        path,
    )


def test_main_runs_two_folds_end_to_end_and_aggregates(tmp_path):
    raw_dir = tmp_path / "raw"
    pitches_dir = tmp_path / "pitches"
    seasons = [2015, 2016, 2017, 2018]
    pitches = _write_fixture(raw_dir, pitches_dir, seasons)

    encoder_checkpoint = tmp_path / "player_encoder_best.pt"
    _write_encoder_checkpoint(encoder_checkpoint, pitches)

    training_config_path = tmp_path / "training_config.yaml"
    training_config_path.write_text(
        "hidden_dim: 16\n"
        "num_layers: 1\n"
        "dropout: 0.0\n"
        "runs_distribution: negative_binomial\n"
        "freeze_encoder: false\n"
        "training_mode: joint\n"
        "stage1_epochs: 1\n"
        "encoder_lr: 0.00001\n"
        "predictor_lr: 0.001\n"
    )

    output_dir = tmp_path / "reports"
    checkpoint_dir = tmp_path / "checkpoints"
    betting_lines_dir = tmp_path / "betting_lines"
    _write_betting_lines_fixture(betting_lines_dir, seasons)

    walk_forward_main(
        [
            "--season-range-start", "2015",
            "--season-range-end", "2018",
            "--train-years", "1",
            "--val-years", "1",
            "--test-years", "1",
            "--training-config", str(training_config_path),
            "--encoder-checkpoint", str(encoder_checkpoint),
            "--pitches-dir", str(pitches_dir),
            "--raw-dir", str(raw_dir),
            "--games-dir", str(tmp_path / "games"),
            "--pitcher-appearances-dir", str(tmp_path / "pitcher_appearances"),
            "--batter-appearances-dir", str(tmp_path / "batter_appearances"),
            "--betting-lines-dir", str(betting_lines_dir),
            "--epochs", "1",
            "--patience", "1",
            "--batch-size", "4",
            "--device", "cpu",
            "--cache-dir", str(tmp_path / "sequence_cache"),
            "--checkpoint-dir", str(checkpoint_dir),
            "--output-dir", str(output_dir),
            "--n-resamples", "20",
            "--seed", "0",
        ]
    )

    fold_summaries = pd.read_csv(output_dir / "fold_summaries.csv")
    assert set(fold_summaries["fold"]) == {0, 1}
    assert set(fold_summaries["method"]) == {
        "GamePredictor", "Logistic regression (ERA/wRC+ proxy)", "Always predict home win", "Vegas closing line",
    }

    fold_bootstrap = pd.read_csv(output_dir / "fold_bootstrap_summary.csv")
    assert len(fold_bootstrap) == 4  # 2 folds x 2 comparisons (GamePredictor vs LR, GamePredictor vs Vegas)
    assert set(fold_bootstrap["comparison"]) == {GP_VS_LR_COMPARISON, GP_VS_VEGAS_COMPARISON}
    assert {"accuracy_diff_mean", "brier_diff_mean", "accuracy_significant_for_gamepredictor"} <= set(fold_bootstrap.columns)

    assert (output_dir / "fold_effects_gamepredictor_vs_lr.png").exists()
    assert (output_dir / "fold_effects_gamepredictor_vs_vegas.png").exists()
    assert (checkpoint_dir / "fold_00_test2017.pt").exists()
    assert (checkpoint_dir / "fold_01_test2018.pt").exists()


def test_resume_skips_already_completed_folds(tmp_path):
    raw_dir = tmp_path / "raw"
    pitches_dir = tmp_path / "pitches"
    seasons = [2015, 2016, 2017, 2018]
    pitches = _write_fixture(raw_dir, pitches_dir, seasons)

    encoder_checkpoint = tmp_path / "player_encoder_best.pt"
    _write_encoder_checkpoint(encoder_checkpoint, pitches)

    training_config_path = tmp_path / "training_config.yaml"
    training_config_path.write_text(
        "hidden_dim: 16\n"
        "num_layers: 1\n"
        "dropout: 0.0\n"
        "runs_distribution: negative_binomial\n"
        "freeze_encoder: false\n"
        "training_mode: joint\n"
        "stage1_epochs: 1\n"
        "encoder_lr: 0.00001\n"
        "predictor_lr: 0.001\n"
    )

    output_dir = tmp_path / "reports"
    checkpoint_dir = tmp_path / "checkpoints"
    betting_lines_dir = tmp_path / "betting_lines"
    _write_betting_lines_fixture(betting_lines_dir, seasons)

    def _base_args(season_range_end):
        return [
            "--season-range-start", "2015",
            "--season-range-end", str(season_range_end),
            "--train-years", "1",
            "--val-years", "1",
            "--test-years", "1",
            "--training-config", str(training_config_path),
            "--encoder-checkpoint", str(encoder_checkpoint),
            "--pitches-dir", str(pitches_dir),
            "--raw-dir", str(raw_dir),
            "--games-dir", str(tmp_path / "games"),
            "--pitcher-appearances-dir", str(tmp_path / "pitcher_appearances"),
            "--batter-appearances-dir", str(tmp_path / "batter_appearances"),
            "--betting-lines-dir", str(betting_lines_dir),
            "--epochs", "1",
            "--patience", "1",
            "--batch-size", "4",
            "--device", "cpu",
            "--cache-dir", str(tmp_path / "sequence_cache"),
            "--checkpoint-dir", str(checkpoint_dir),
            "--output-dir", str(output_dir),
            "--n-resamples", "20",
            "--seed", "0",
        ]

    # first run: only season range 2015-2017 exists -> exactly fold 0
    walk_forward_main(_base_args(2017))
    fold_00_checkpoint = checkpoint_dir / "fold_00_test2017.pt"
    assert fold_00_checkpoint.exists()
    first_run_mtime = fold_00_checkpoint.stat().st_mtime

    fold_bootstrap = pd.read_csv(output_dir / "fold_bootstrap_summary.csv")
    assert set(fold_bootstrap["fold"]) == {0}

    # second run: full range (folds 0 and 1) with --resume -- fold 0 must not be retrained
    walk_forward_main(_base_args(2018) + ["--resume"])

    assert fold_00_checkpoint.stat().st_mtime == first_run_mtime  # untouched

    fold_bootstrap = pd.read_csv(output_dir / "fold_bootstrap_summary.csv")
    assert set(fold_bootstrap["fold"]) == {0, 1}

    fold_summaries = pd.read_csv(output_dir / "fold_summaries.csv")
    assert set(fold_summaries["fold"]) == {0, 1}

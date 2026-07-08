from dataclasses import asdict

import pandas as pd
import pytest
import torch
import yaml

from src.data.build_features import build_season_pitches_from_frame
from src.data.sequence_dataset import PlayerPitchSequenceDataset
from src.data.statcast_common import (
    TRAIN_SEASON_RANGE,
    VAL_SEASONS,
    build_pitch_frame_from_raw,
    write_partitioned,
)
from src.models.player_encoder import PlayerEncoder, PlayerEncoderConfig
from src.training.train_game_predictor import (
    GamePredictorTrainingConfig,
    _build_context_features,
    _compute_rest_day_stats,
    _encoder_trainable_this_epoch,
    _pad_player_sequences,
    build_random_encoder,
    main as train_main,
)


def test_training_config_rejects_invalid_training_mode():
    with pytest.raises(ValueError):
        GamePredictorTrainingConfig(training_mode="bogus")


@pytest.mark.parametrize(
    "freeze_encoder,training_mode,stage1_epochs,epoch,expected",
    [
        (True, "joint", 5, 1, False),
        (True, "two_stage", 5, 10, False),  # freeze_encoder overrides two_stage entirely
        (False, "joint", 5, 1, True),
        (False, "two_stage", 3, 3, False),  # still in stage 1
        (False, "two_stage", 3, 4, True),  # stage 2 begins
    ],
)
def test_encoder_trainable_this_epoch(freeze_encoder, training_mode, stage1_epochs, epoch, expected):
    config = GamePredictorTrainingConfig(
        freeze_encoder=freeze_encoder, training_mode=training_mode, stage1_epochs=stage1_epochs
    )
    assert _encoder_trainable_this_epoch(epoch, config) is expected


def test_compute_rest_day_stats_ignores_missing_first_starts():
    games = pd.DataFrame(
        {
            "home_starter_rest_days": [4.0, 6.0, float("nan")],
            "away_starter_rest_days": [5.0, float("nan"), 5.0],
        }
    )
    mean, std = _compute_rest_day_stats(games)
    assert mean == pytest.approx(pd.Series([4.0, 6.0, 5.0, 5.0]).mean())
    assert std > 0


def test_build_context_features_flags_missing_rest_days():
    month = torch.tensor([4.0, 10.0])
    home_rest = torch.tensor([5.0, float("nan")])
    away_rest = torch.tensor([float("nan"), 3.0])
    post_humidor = torch.tensor([0.0, 1.0])

    context = _build_context_features(
        month, home_rest, away_rest, rest_mean=5.0, rest_std=2.0, post_humidor=post_humidor
    )

    assert context.shape == (2, 7)
    # column order: month_sin, month_cos, home_norm, home_missing, away_norm, away_missing, post_humidor
    assert context[0, 3].item() == 0.0  # home rest present for game 0
    assert context[0, 5].item() == 1.0  # away rest missing for game 0
    assert context[1, 3].item() == 1.0  # home rest missing for game 1
    assert context[1, 5].item() == 0.0  # away rest present for game 1
    assert context[0, 6].item() == 0.0  # pre-humidor game 0
    assert context[1, 6].item() == 1.0  # post-humidor game 1
    assert not torch.isnan(context).any()  # missing rest days must not leak NaN into the tensor


def test_pad_player_sequences_handles_empty_list():
    padded = _pad_player_sequences([])
    assert padded["continuous"].shape == (0, 1, 4)
    assert padded["has_history"].shape == (0,)


def _raw_row(pitcher, batter, game_date, at_bat_number, pitch_number, inning_topbot, home_team, away_team, home_score, away_score, season):
    return {
        "pitcher": pitcher,
        "batter": batter,
        "game_date": game_date,
        "game_pk": season,  # one game per season in this fixture, so season doubles as a unique game_pk
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
    """One game per season across the full train+val range (2015-2023) --
    load_game_split() builds game tables for that whole hardcoded range, so
    every season in it needs a raw file. Each game: home starter 100 faces 9
    away batters, away starter 200 faces 9 home batters, DET beats CLE 5-3.
    No relievers, so every bullpen is empty -- exercises the empty-bullpen
    path (PlayerSetPooler's learned empty-set embedding) end-to-end."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []
    seasons = list(range(TRAIN_SEASON_RANGE[0], TRAIN_SEASON_RANGE[1] + 1)) + list(VAL_SEASONS)
    for season in seasons:
        date = f"{season}-04-01"
        rows = [
            _raw_row(100, 101 + i, date, i + 1, 1, "Top", "DET", "CLE", 5, 3, season) for i in range(9)
        ] + [
            _raw_row(200, 1 + i, date, 10 + i, 1, "Bot", "DET", "CLE", 5, 3, season) for i in range(9)
        ]
        pd.DataFrame(rows).to_parquet(raw_dir / f"statcast_{season}.parquet")
        all_rows.extend(rows)

    raw_all = pd.DataFrame(all_rows)
    pitches = build_season_pitches_from_frame(build_pitch_frame_from_raw(raw_all))
    write_partitioned(pitches, pitches_dir)
    return pitches


def _write_encoder_checkpoint(path, pitches):
    train_pitches = pitches[pitches["season"].between(*TRAIN_SEASON_RANGE) & pitches["is_valid"]]
    continuous_stats = PlayerPitchSequenceDataset._compute_continuous_stats(train_pitches)

    config = PlayerEncoderConfig(hidden_size=8, num_layers=1, num_heads=2, dropout=0.0, feedforward_dim=16, max_seq_len=5)
    encoder = PlayerEncoder(config)
    torch.save(
        {
            "encoder_state_dict": encoder.state_dict(),
            "config": asdict(config),
            "continuous_stats": continuous_stats,
            "epoch": 1,
            "val_loss": 0.1,
            "val_accuracy": 0.5,
        },
        path,
    )


def test_main_runs_end_to_end_and_writes_log_and_checkpoint(tmp_path):
    raw_dir = tmp_path / "raw"
    pitches_dir = tmp_path / "pitches"
    pitches = _write_fixture(raw_dir, pitches_dir)

    encoder_checkpoint = tmp_path / "player_encoder_best.pt"
    _write_encoder_checkpoint(encoder_checkpoint, pitches)

    training_config_path = tmp_path / "training_config.yaml"
    training_config_path.write_text(
        yaml.dump(
            {
                "hidden_dim": 16,
                "num_layers": 1,
                "dropout": 0.0,
                "runs_distribution": "negative_binomial",
                "freeze_encoder": False,
                "training_mode": "two_stage",
                "stage1_epochs": 1,
                "encoder_lr": 1e-5,
                "predictor_lr": 1e-3,
            }
        )
    )

    log_dir = tmp_path / "logs"
    checkpoint_dir = tmp_path / "checkpoints"

    train_main(
        [
            "--training-config", str(training_config_path),
            "--encoder-checkpoint", str(encoder_checkpoint),
            "--pitches-dir", str(pitches_dir),
            "--raw-dir", str(raw_dir),
            "--games-dir", str(tmp_path / "games"),
            "--pitcher-appearances-dir", str(tmp_path / "pitcher_appearances"),
            "--batter-appearances-dir", str(tmp_path / "batter_appearances"),
            "--epochs", "2",
            "--batch-size", "4",
            "--log-dir", str(log_dir),
            "--checkpoint-dir", str(checkpoint_dir),
            "--device", "cpu",
            "--cache-dir", str(tmp_path / "sequence_cache"),
        ]
    )

    log_path = log_dir / "train_game_predictor.csv"
    assert log_path.exists()
    log_lines = log_path.read_text().strip().splitlines()
    assert log_lines[0] == (
        "epoch,encoder_trainable,train_loss,train_brier,train_home_mae,train_away_mae,"
        "val_loss,val_brier,val_home_mae,val_away_mae"
    )
    assert len(log_lines) == 3  # header + 2 epochs
    # epoch 1 is stage 1 (encoder frozen), epoch 2 is stage 2 (unfrozen)
    assert log_lines[1].split(",")[1] == "False"
    assert log_lines[2].split(",")[1] == "True"

    checkpoint_path = checkpoint_dir / "game_predictor_best.pt"
    assert checkpoint_path.exists()
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    assert set(checkpoint.keys()) >= {
        "game_predictor_state_dict", "bullpen_pooler_state_dict", "lineup_pooler_state_dict",
        "encoder_config", "predictor_config", "pooler_config", "continuous_stats", "rest_day_stats",
        "epoch", "val_loss", "val_brier", "val_home_mae", "val_away_mae",
    }

    # the sequence cache should have been warmed (one file per player queried)
    cache_dir = tmp_path / "sequence_cache"
    assert (cache_dir / "pitcher").exists()
    assert any((cache_dir / "pitcher").iterdir())


def test_build_random_encoder_is_not_pretrained_and_has_matching_continuous_stats(tmp_path):
    raw_dir = tmp_path / "raw"
    pitches_dir = tmp_path / "pitches"
    pitches = _write_fixture(raw_dir, pitches_dir)

    tiny_encoder_config = tmp_path / "encoder_config.yaml"
    tiny_encoder_config.write_text(
        yaml.dump({"hidden_size": 8, "num_layers": 1, "num_heads": 2, "dropout": 0.0, "feedforward_dim": 16, "max_seq_len": 5})
    )

    encoder, continuous_stats = build_random_encoder(tiny_encoder_config, pitches_dir)

    assert encoder.config.hidden_size == 8
    train_pitches = pitches[pitches["season"].between(*TRAIN_SEASON_RANGE) & pitches["is_valid"]]
    expected_stats = PlayerPitchSequenceDataset._compute_continuous_stats(train_pitches)
    assert continuous_stats == expected_stats

    # two independent calls must NOT produce identical weights -- that would
    # mean it's silently reusing/sharing state instead of actually random init.
    encoder2, _ = build_random_encoder(tiny_encoder_config, pitches_dir)
    assert not torch.equal(encoder.continuous_proj.weight, encoder2.continuous_proj.weight)


def test_main_with_no_pretrained_encoder_trains_without_a_checkpoint(tmp_path):
    raw_dir = tmp_path / "raw"
    pitches_dir = tmp_path / "pitches"
    _write_fixture(raw_dir, pitches_dir)

    tiny_encoder_config = tmp_path / "encoder_config.yaml"
    tiny_encoder_config.write_text(
        yaml.dump({"hidden_size": 8, "num_layers": 1, "num_heads": 2, "dropout": 0.0, "feedforward_dim": 16, "max_seq_len": 5})
    )

    training_config_path = tmp_path / "training_config.yaml"
    training_config_path.write_text(
        yaml.dump(
            {
                "hidden_dim": 16,
                "num_layers": 1,
                "dropout": 0.0,
                "runs_distribution": "negative_binomial",
                "freeze_encoder": False,
                "training_mode": "two_stage",
                "stage1_epochs": 1,
                "encoder_lr": 1e-5,
                "predictor_lr": 1e-3,
            }
        )
    )

    log_dir = tmp_path / "logs"
    checkpoint_dir = tmp_path / "checkpoints"

    # deliberately point --encoder-checkpoint at a file that doesn't exist --
    # --no-pretrained-encoder must mean it's never even opened.
    train_main(
        [
            "--no-pretrained-encoder",
            "--encoder-config", str(tiny_encoder_config),
            "--encoder-checkpoint", str(tmp_path / "does_not_exist.pt"),
            "--training-config", str(training_config_path),
            "--pitches-dir", str(pitches_dir),
            "--raw-dir", str(raw_dir),
            "--games-dir", str(tmp_path / "games"),
            "--pitcher-appearances-dir", str(tmp_path / "pitcher_appearances"),
            "--batter-appearances-dir", str(tmp_path / "batter_appearances"),
            "--epochs", "1",
            "--batch-size", "4",
            "--log-dir", str(log_dir),
            "--checkpoint-dir", str(checkpoint_dir),
            "--device", "cpu",
            "--cache-dir", str(tmp_path / "sequence_cache"),
        ]
    )

    checkpoint_path = checkpoint_dir / "game_predictor_best.pt"
    assert checkpoint_path.exists()
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    assert checkpoint["encoder_config"]["hidden_size"] == 8

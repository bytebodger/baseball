import pytest
import yaml
import torch
import pandas as pd

import src.training.pretrain_encoder as pretrain_encoder_module
from src.data.sequence_dataset import PlayerPitchSequenceDataset
from src.data.statcast_common import build_pitch_frame_from_raw, write_partitioned
from src.data.build_features import build_season_pitches_from_frame
from src.models.player_encoder import PlayerEncoder, PlayerEncoderConfig
from src.training.pretrain_encoder import (
    NextPitchDataset,
    NextPitchPredictor,
    collate_next_pitch_batch,
    naive_baseline_metrics,
    main as pretrain_main,
)


def _raw_row(pitcher, batter, game_date, at_bat_number, pitch_number, events=None, description="ball", season=2024):
    return {
        "pitcher": pitcher,
        "batter": batter,
        "game_date": game_date,
        "game_pk": 1,
        "at_bat_number": at_bat_number,
        "pitch_number": pitch_number,
        "pitch_type": "FF",
        "release_speed": 90.0 + pitch_number,
        "release_spin_rate": 2200 + pitch_number,
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
        "inning": 1,
        "stand": "R",
        "p_throws": "L",
        "home_team": "DET",
        "away_team": "CLE",
        "game_year": season,
        "events": events,
        "description": description,
    }


def _clean_pitches_frame() -> pd.DataFrame:
    """Two pitchers: 100 has 8 pitches (to test truncation with max_seq_len=5),
    200 has a single pitch (to test the zero-history / group-start case)."""
    rows = [
        _raw_row(100, 1, f"2024-04-{i + 1:02d}", i + 1, 1) for i in range(8)
    ] + [_raw_row(200, 2, "2024-04-01", 1, 1)]
    raw = pd.DataFrame(rows)
    return build_pitch_frame_from_raw(raw)


def test_first_pitch_of_a_pitcher_has_zero_history():
    pitches = _clean_pitches_frame()
    stats = PlayerPitchSequenceDataset._compute_continuous_stats(pitches)
    dataset = NextPitchDataset(pitches, max_seq_len=5, continuous_stats=stats)

    sorted_pitches = pitches.sort_values(["pitcher_id", "game_date", "at_bat_number", "pitch_number"]).reset_index(
        drop=True
    )
    first_idx_pitcher_100 = sorted_pitches.index[sorted_pitches["pitcher_id"] == 100][0]

    sample = dataset[first_idx_pitcher_100]
    assert sample["has_history"] is False
    assert sample["length"] == 0


def test_truncates_history_to_max_seq_len_keeping_most_recent():
    pitches = _clean_pitches_frame()
    stats = PlayerPitchSequenceDataset._compute_continuous_stats(pitches)
    dataset = NextPitchDataset(pitches, max_seq_len=5, continuous_stats=stats)

    sorted_pitches = pitches.sort_values(["pitcher_id", "game_date", "at_bat_number", "pitch_number"]).reset_index(
        drop=True
    )
    last_idx_pitcher_100 = sorted_pitches.index[sorted_pitches["pitcher_id"] == 100][-1]

    sample = dataset[last_idx_pitcher_100]
    assert sample["has_history"] is True
    assert sample["length"] == 5  # capped at max_seq_len even though 7 prior pitches exist


def test_target_is_the_pitch_own_outcome():
    pitches = _clean_pitches_frame()
    stats = PlayerPitchSequenceDataset._compute_continuous_stats(pitches)
    dataset = NextPitchDataset(pitches, max_seq_len=5, continuous_stats=stats)

    sorted_pitches = pitches.sort_values(["pitcher_id", "game_date", "at_bat_number", "pitch_number"]).reset_index(
        drop=True
    )
    idx = sorted_pitches.index[sorted_pitches["pitcher_id"] == 200][0]
    sample = dataset[idx]

    from src.data.sequence_dataset import OUTCOME_INDEX

    assert sample["target"] == OUTCOME_INDEX[sorted_pitches.loc[idx, "outcome"]]


def test_naive_baseline_metrics_uses_majority_class_frequency_as_accuracy():
    val_df = pd.DataFrame({"outcome": ["ball", "ball", "ball", "called_strike"]})

    accuracy, cross_entropy, majority_class = naive_baseline_metrics(val_df)

    assert majority_class == "ball"
    assert accuracy == 0.75
    assert cross_entropy > 0
    # Predicting the wrong class costs far more than predicting the right one,
    # so a baseline that's wrong 25% of the time should log-loss noticeably
    # worse than a nearly-confident-and-correct model would.
    assert cross_entropy > 1.0


def test_naive_baseline_metrics_is_perfect_when_val_set_is_single_class():
    val_df = pd.DataFrame({"outcome": ["strikeout", "strikeout", "strikeout"]})

    accuracy, cross_entropy, majority_class = naive_baseline_metrics(val_df)

    assert majority_class == "strikeout"
    assert accuracy == 1.0
    assert cross_entropy == pytest.approx(0.0, abs=1e-6)


def test_collate_pads_and_masks_mixed_lengths():
    pitches = _clean_pitches_frame()
    stats = PlayerPitchSequenceDataset._compute_continuous_stats(pitches)
    dataset = NextPitchDataset(pitches, max_seq_len=5, continuous_stats=stats)

    batch = [dataset[i] for i in range(len(dataset))]
    inputs, targets = collate_next_pitch_batch(batch)

    max_len = max(sample["length"] for sample in batch)
    assert inputs["continuous"].shape == (len(batch), max_len, 4)
    assert targets.shape == (len(batch),)

    zero_history_positions = [i for i, s in enumerate(batch) if not s["has_history"]]
    for i in zero_history_positions:
        assert inputs["padding_mask"][i].all()  # fully masked
        assert not inputs["has_history"][i]


def test_next_pitch_predictor_output_shape():
    config = PlayerEncoderConfig(hidden_size=8, num_layers=1, num_heads=2, dropout=0.0, feedforward_dim=16, max_seq_len=5)
    model = NextPitchPredictor(config)

    pitches = _clean_pitches_frame()
    stats = PlayerPitchSequenceDataset._compute_continuous_stats(pitches)
    dataset = NextPitchDataset(pitches, max_seq_len=5, continuous_stats=stats)
    batch = [dataset[i] for i in range(len(dataset))]
    inputs, targets = collate_next_pitch_batch(batch)

    logits = model(inputs)
    from src.data.sequence_dataset import OUTCOME_VOCAB

    assert logits.shape == (len(batch), len(OUTCOME_VOCAB))


def _write_fake_processed_dataset(base_dir):
    """Build a tiny but real processed pitches dataset (via the real
    build_features pipeline) spanning a train season (2015) and a val season
    (2023), then write it partitioned by season -- exactly the on-disk layout
    pretrain_encoder.main() expects."""
    train_rows = [_raw_row(100, 1, f"2015-04-{i + 1:02d}", i + 1, 1, season=2015) for i in range(20)]
    val_rows = [_raw_row(100, 1, f"2023-04-{i + 1:02d}", i + 1, 1, season=2023) for i in range(8)]
    raw = pd.DataFrame(train_rows + val_rows)

    pitches = build_season_pitches_from_frame(build_pitch_frame_from_raw(raw))
    write_partitioned(pitches, base_dir)


def test_main_runs_end_to_end_and_writes_log_and_checkpoint(tmp_path):
    pitches_dir = tmp_path / "pitches"
    _write_fake_processed_dataset(pitches_dir)

    config_path = tmp_path / "tiny_config.yaml"
    config_path.write_text(
        yaml.dump(
            {"hidden_size": 8, "num_layers": 1, "num_heads": 2, "dropout": 0.0, "feedforward_dim": 16, "max_seq_len": 5}
        )
    )

    log_dir = tmp_path / "logs"
    checkpoint_dir = tmp_path / "checkpoints"

    pretrain_main(
        [
            "--config", str(config_path),
            "--pitches-dir", str(pitches_dir),
            "--epochs", "1",
            "--batch-size", "4",
            "--log-dir", str(log_dir),
            "--checkpoint-dir", str(checkpoint_dir),
            "--device", "cpu",
        ]
    )

    log_path = log_dir / "pretrain_encoder.csv"
    assert log_path.exists()
    log_lines = log_path.read_text().strip().splitlines()
    assert log_lines[0] == "epoch,train_loss,train_accuracy,val_loss,val_accuracy"
    assert len(log_lines) == 2  # header + 1 epoch

    checkpoint_path = checkpoint_dir / "player_encoder_best.pt"
    assert checkpoint_path.exists()
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    assert checkpoint["epoch"] == 1
    assert set(checkpoint.keys()) >= {
        "encoder_state_dict", "classifier_state_dict", "config", "continuous_stats", "val_loss", "val_accuracy",
    }

    # the saved encoder weights should load cleanly into a fresh encoder built from the same config
    config = PlayerEncoderConfig(**checkpoint["config"])
    encoder = PlayerEncoder(config)
    encoder.load_state_dict(checkpoint["encoder_state_dict"])


def test_early_stopping_halts_training_after_patience_epochs_without_improvement(tmp_path, monkeypatch):
    pitches_dir = tmp_path / "pitches"
    _write_fake_processed_dataset(pitches_dir)

    config_path = tmp_path / "tiny_config.yaml"
    config_path.write_text(
        yaml.dump(
            {"hidden_size": 8, "num_layers": 1, "num_heads": 2, "dropout": 0.0, "feedforward_dim": 16, "max_seq_len": 5}
        )
    )

    # Deterministic val_loss trajectory: improves at epoch 1 and 2, then sits
    # flat for 4 consecutive epochs (3-6) -- with patience=4 that should stop
    # training right after epoch 6, even though --epochs allows up to 20.
    val_losses = iter([1.0, 0.9, 0.95, 0.95, 0.95, 0.95, 0.5, 0.5])

    def fake_run_epoch(model, loader, device, criterion, optimizer=None, scaler=None, use_amp=False):
        if optimizer is not None:
            return 1.0, 0.5  # train_loss, train_accuracy (unused by this test)
        return next(val_losses), 0.5

    monkeypatch.setattr(pretrain_encoder_module, "run_epoch", fake_run_epoch)

    log_dir = tmp_path / "logs"
    checkpoint_dir = tmp_path / "checkpoints"

    pretrain_main(
        [
            "--config", str(config_path),
            "--pitches-dir", str(pitches_dir),
            "--epochs", "20",
            "--patience", "4",
            "--batch-size", "4",
            "--log-dir", str(log_dir),
            "--checkpoint-dir", str(checkpoint_dir),
            "--device", "cpu",
        ]
    )

    log_lines = (log_dir / "pretrain_encoder.csv").read_text().strip().splitlines()
    assert len(log_lines) == 1 + 6  # header + 6 epochs, not the full 20

    checkpoint = torch.load(checkpoint_dir / "player_encoder_best.pt", weights_only=False)
    assert checkpoint["epoch"] == 2  # best val_loss (0.9) was at epoch 2
    assert checkpoint["val_loss"] == 0.9

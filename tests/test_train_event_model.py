from dataclasses import asdict

import pandas as pd
import torch
import yaml

from src.data.build_features import build_season_pitches_from_frame
from src.data.event_embedding_cache import precompute_and_cache_embeddings
from src.data.statcast_common import TRAIN_SEASON_RANGE, VAL_SEASONS, build_pitch_frame_from_raw, read_partitioned, write_partitioned
from src.models.long_history_encoder import CareerEncoder, CareerEncoderConfig, ChunkEncoder, ChunkEncoderConfig, LongHistoryEncoder
from src.training.train_event_model import EventTrainingConfig, main as train_main


def _raw_row(pitcher, batter, game_date, at_bat, pitch_num, balls, strikes, outs, on1, on2, on3,
             home_score, away_score, tto, events, description, season, inning_topbot="Top"):
    return {
        "pitcher": pitcher,
        "batter": batter,
        "game_date": game_date,
        "game_pk": season,  # one game per season in this fixture
        "game_year": season,
        "home_team": "DET",
        "away_team": "CLE",
        "inning": 1,
        "inning_topbot": inning_topbot,
        "at_bat_number": at_bat,
        "pitch_number": pitch_num,
        "pitch_type": "FF",
        "release_speed": 90.0,
        "release_spin_rate": 2200,
        "spin_rate_deprecated": None,
        "plate_x": 0.1,
        "plate_z": 2.2,
        "balls": balls,
        "strikes": strikes,
        "outs_when_up": outs,
        "on_1b": on1,
        "on_2b": on2,
        "on_3b": on3,
        "home_score": home_score,
        "away_score": away_score,
        "n_thruorder_pitcher": tto + 1,
        "stand": "R",
        "p_throws": "L",
        "events": events,
        "description": description,
    }


def _write_fixture(raw_dir, pitches_dir):
    """One game per season across the full train+val range -- enough
    distinct (pitcher, date)/(batter, date) pairs and outcome diversity to
    exercise EventDataset/EventModel/train_event_model end-to-end."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []
    seasons = list(range(TRAIN_SEASON_RANGE[0], TRAIN_SEASON_RANGE[1] + 1)) + list(VAL_SEASONS)
    for season in seasons:
        date = f"{season}-04-01"
        rows = [
            _raw_row(100, 101, date, 1, 1, 0, 0, 0, None, None, None, 0, 0, 0, None, "ball", season),
            _raw_row(100, 101, date, 1, 2, 1, 1, 0, None, None, None, 0, 0, 0, None, "called_strike", season),
            _raw_row(100, 102, date, 2, 1, 0, 0, 1, 555, None, None, 0, 0, 1, "single", "hit_into_play", season),
            _raw_row(100, 103, date, 3, 1, 3, 2, 2, 555, 556, 557, 1, 0, 0, "strikeout", "swinging_strike", season),
            _raw_row(200, 101, date, 4, 1, 0, 0, 0, None, None, None, 1, 0, 0, "home_run", "hit_into_play", season),
        ]
        pd.DataFrame(rows).to_parquet(raw_dir / f"statcast_{season}.parquet")
        all_rows.extend(rows)

    raw_all = pd.DataFrame(all_rows)
    pitches = build_season_pitches_from_frame(build_pitch_frame_from_raw(raw_all))
    write_partitioned(pitches, pitches_dir)
    return pitches


def _write_embedding_cache(pitches, cache_dir):
    chunk_config = ChunkEncoderConfig(hidden_size=8, num_layers=1, num_heads=2, dropout=0.0, feedforward_dim=16, max_seq_len=10)
    career_config = CareerEncoderConfig(hidden_size=8, num_layers=1, num_heads=2, dropout=0.0, feedforward_dim=16, max_chunks=6)
    torch.manual_seed(0)
    encoder = LongHistoryEncoder(ChunkEncoder(chunk_config), CareerEncoder(career_config))
    precompute_and_cache_embeddings(
        pitches, encoder, cache_dir, career_config.max_chunks, chunk_config.max_seq_len, device=torch.device("cpu"), batch_size=4
    )
    return career_config.hidden_size


def _write_training_config(path):
    path.write_text(
        yaml.dump(
            {
                "hidden_dim": 16,
                "num_layers": 1,
                "dropout": 0.0,
                "matchup_embed_dim": 4,
                "park_factor_embed_dim": 4,
                "park_factor_rolling_years": 3,
                "lr": 1e-3,
            }
        )
    )


def test_training_config_to_event_model_config_maps_fields():
    config = EventTrainingConfig(hidden_dim=32, num_layers=3, dropout=0.2, matchup_embed_dim=6, park_factor_embed_dim=6, lr=5e-4)
    model_config = config.to_event_model_config(player_embed_dim=128, include_context=True)
    assert model_config.hidden_dim == 32
    assert model_config.num_layers == 3
    assert model_config.player_embed_dim == 128
    assert model_config.matchup_embed_dim == 6
    assert model_config.park_factor_embed_dim == 6
    assert model_config.include_context is True


def test_main_runs_end_to_end_with_context_and_writes_log_and_checkpoint(tmp_path):
    raw_dir = tmp_path / "raw"
    pitches_dir = tmp_path / "pitches"
    pitches = _write_fixture(raw_dir, pitches_dir)

    cache_dir = tmp_path / "embedding_cache"
    _write_embedding_cache(read_partitioned(pitches_dir), cache_dir)

    training_config_path = tmp_path / "training_config.yaml"
    _write_training_config(training_config_path)

    log_dir = tmp_path / "logs"
    checkpoint_dir = tmp_path / "checkpoints"

    train_main(
        [
            "--training-config", str(training_config_path),
            "--pitches-dir", str(pitches_dir),
            "--embedding-cache-dir", str(cache_dir),
            "--epochs", "2",
            "--batch-size", "4",
            "--log-dir", str(log_dir),
            "--checkpoint-dir", str(checkpoint_dir),
            "--device", "cpu",
        ]
    )

    log_path = log_dir / "train_event_model_full.csv"
    assert log_path.exists()
    lines = log_path.read_text().strip().splitlines()
    assert lines[0] == "epoch,train_loss,train_accuracy,val_loss,val_accuracy"
    assert len(lines) == 3  # header + 2 epochs

    checkpoint_path = checkpoint_dir / "event_model_full_best.pt"
    assert checkpoint_path.exists()
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    assert set(checkpoint.keys()) >= {
        "model_state_dict", "model_config", "park_factor_config", "situational_stats", "epoch", "val_loss", "val_accuracy",
    }
    assert checkpoint["model_config"]["include_context"] is True


def test_main_with_no_context_writes_a_separate_log_and_checkpoint(tmp_path):
    raw_dir = tmp_path / "raw"
    pitches_dir = tmp_path / "pitches"
    _write_fixture(raw_dir, pitches_dir)

    cache_dir = tmp_path / "embedding_cache"
    _write_embedding_cache(read_partitioned(pitches_dir), cache_dir)

    training_config_path = tmp_path / "training_config.yaml"
    _write_training_config(training_config_path)

    log_dir = tmp_path / "logs"
    checkpoint_dir = tmp_path / "checkpoints"

    train_main(
        [
            "--no-context",
            "--training-config", str(training_config_path),
            "--pitches-dir", str(pitches_dir),
            "--embedding-cache-dir", str(cache_dir),
            "--epochs", "1",
            "--batch-size", "4",
            "--log-dir", str(log_dir),
            "--checkpoint-dir", str(checkpoint_dir),
            "--device", "cpu",
        ]
    )

    log_path = log_dir / "train_event_model_no_context.csv"
    assert log_path.exists()
    checkpoint_path = checkpoint_dir / "event_model_no_context_best.pt"
    assert checkpoint_path.exists()
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    assert checkpoint["model_config"]["include_context"] is False

    # Running the default (with-context) config too must not clobber this file.
    assert not (log_dir / "train_event_model_full.csv").exists()

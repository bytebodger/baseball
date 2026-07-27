import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from src.data.build_features import build_season_pitches_from_frame
from src.data.contact_quality import build_contact_quality_history, load_raw_batted_balls, save_contact_quality_histories
from src.data.event_dataset import EventBatchCollator, EventDataset, compute_contact_quality_stats, compute_situational_stats
from src.data.event_embedding_cache import EmbeddingCache, precompute_and_cache_embeddings
from src.data.park_factors import ParkFactorConfig, ParkFactorEmbedding, compute_league_rates, compute_park_factors
from src.data.sequence_dataset import OUTCOME_VOCAB
from src.data.statcast_common import TRAIN_SEASON_RANGE, VAL_SEASONS, build_pitch_frame_from_raw, read_partitioned, write_partitioned
from src.evaluation.event_model_calibration import compute_marginal_calibration
from src.evaluation.event_model_calibration import main as calibration_main
from src.models.event_model import EventModel, EventModelConfig
from src.models.long_history_encoder import CareerEncoder, CareerEncoderConfig, ChunkEncoder, ChunkEncoderConfig, LongHistoryEncoder
from src.training.train_event_model import main as train_main


def _raw_row(pitcher, batter, game_date, at_bat, pitch_num, balls, strikes, outs, on1, on2, on3,
             home_score, away_score, tto, events, description, season, inning_topbot="Top", launch_speed=None, type_flag="S"):
    return {
        "pitcher": pitcher,
        "batter": batter,
        "game_date": game_date,
        "game_pk": season,
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
        "launch_speed": launch_speed,
        "type": type_flag,
    }


def _write_fixture(raw_dir, pitches_dir):
    raw_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []
    seasons = list(range(TRAIN_SEASON_RANGE[0], TRAIN_SEASON_RANGE[1] + 1)) + list(VAL_SEASONS)
    for season in seasons:
        date = f"{season}-04-01"
        rows = [
            _raw_row(100, 101, date, 1, 1, 0, 0, 0, None, None, None, 0, 0, 0, None, "ball", season, type_flag="B"),
            _raw_row(100, 101, date, 1, 2, 1, 1, 0, None, None, None, 0, 0, 0, None, "called_strike", season, type_flag="S"),
            _raw_row(100, 102, date, 2, 1, 0, 0, 1, 555, None, None, 0, 0, 1, "single", "hit_into_play", season, launch_speed=95.0, type_flag="X"),
            _raw_row(100, 103, date, 3, 1, 3, 2, 2, 555, 556, 557, 1, 0, 0, "strikeout", "swinging_strike", season, type_flag="S"),
            _raw_row(200, 101, date, 4, 1, 0, 0, 0, None, None, None, 1, 0, 0, "home_run", "hit_into_play", season, launch_speed=105.0, type_flag="X"),
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


def _write_contact_quality_checkpoint(raw_dir, checkpoint_path):
    batted_balls = load_raw_batted_balls(raw_dir=raw_dir)
    pitcher_history = build_contact_quality_history(batted_balls, "pitcher_id")
    batter_history = build_contact_quality_history(batted_balls, "batter_id")
    save_contact_quality_histories(pitcher_history, batter_history, checkpoint_path)


def _write_training_config(path):
    import yaml
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


def _train_a_real_checkpoint(tmp_path):
    """Produces a genuine event_model_full_best.pt via the real
    train_event_model.main() -- the calibration script's whole job is to
    correctly reload and re-evaluate exactly this kind of checkpoint, so
    testing against a hand-built one (rather than a real training run's
    output) would risk missing a reconstruction mismatch (e.g. park-factor
    vocab) that only shows up against the real thing."""
    raw_dir = tmp_path / "raw"
    pitches_dir = tmp_path / "pitches"
    _write_fixture(raw_dir, pitches_dir)

    cache_dir = tmp_path / "embedding_cache"
    _write_embedding_cache(read_partitioned(pitches_dir), cache_dir)

    contact_quality_checkpoint = tmp_path / "contact_quality.pkl"
    _write_contact_quality_checkpoint(raw_dir, contact_quality_checkpoint)

    training_config_path = tmp_path / "training_config.yaml"
    _write_training_config(training_config_path)

    checkpoint_dir = tmp_path / "checkpoints"
    train_main(
        [
            "--training-config", str(training_config_path),
            "--pitches-dir", str(pitches_dir),
            "--embedding-cache-dir", str(cache_dir),
            "--contact-quality-checkpoint", str(contact_quality_checkpoint),
            "--epochs", "1",
            "--batch-size", "4",
            "--log-dir", str(tmp_path / "logs"),
            "--checkpoint-dir", str(checkpoint_dir),
            "--device", "cpu",
        ]
    )
    return {
        "checkpoint_path": checkpoint_dir / "event_model_full_best.pt",
        "pitches_dir": pitches_dir,
        "embedding_cache_dir": cache_dir,
        "contact_quality_checkpoint": contact_quality_checkpoint,
    }


def test_main_runs_end_to_end_and_reports_every_outcome_category(tmp_path, caplog):
    paths = _train_a_real_checkpoint(tmp_path)

    with caplog.at_level("INFO"):
        calibration_main(
            [
                "--checkpoint-path", str(paths["checkpoint_path"]),
                "--pitches-dir", str(paths["pitches_dir"]),
                "--embedding-cache-dir", str(paths["embedding_cache_dir"]),
                "--contact-quality-checkpoint", str(paths["contact_quality_checkpoint"]),
                "--batch-size", "4",
                "--device", "cpu",
            ]
        )

    assert "Marginal calibration over" in caplog.text
    for cat in OUTCOME_VOCAB:
        assert cat in caplog.text


def test_main_raises_on_a_train_season_range_that_does_not_match_the_checkpoint(tmp_path):
    """Regression guard for the exact fragility --train-season-start's help
    text warns about: passing a season range that doesn't match what the
    checkpoint was actually trained with rebuilds a park-factor vocab of a
    different size than the trained embedding table in model_state_dict,
    which load_state_dict must refuse rather than silently misalign."""
    paths = _train_a_real_checkpoint(tmp_path)

    with pytest.raises(RuntimeError):
        calibration_main(
            [
                "--checkpoint-path", str(paths["checkpoint_path"]),
                "--pitches-dir", str(paths["pitches_dir"]),
                "--embedding-cache-dir", str(paths["embedding_cache_dir"]),
                "--contact-quality-checkpoint", str(paths["contact_quality_checkpoint"]),
                "--train-season-start", str(TRAIN_SEASON_RANGE[0] + 1),
                "--batch-size", "4",
                "--device", "cpu",
            ]
        )


def test_compute_marginal_calibration_predicted_and_observed_probabilities_each_sum_to_one():
    """Direct unit test of the core arithmetic, independent of the full
    checkpoint-loading pipeline: both the predicted-average-probability
    distribution and the observed-frequency distribution must each be a
    valid probability distribution over OUTCOME_VOCAB (sums to 1), since
    every row contributes exactly one softmax distribution (sums to 1) and
    exactly one observed category."""
    pitcher_embed_dim = 8
    config = EventModelConfig(
        player_embed_dim=pitcher_embed_dim, matchup_embed_dim=4, park_factor_embed_dim=4,
        situational_dim=13, hidden_dim=16, num_layers=1, dropout=0.0, include_context=False,
    )
    torch.manual_seed(0)
    model = EventModel(config, park_factor_embedding=None)

    n_rows = 37
    dataset = [
        {
            "pitcher_embedding": torch.randn(pitcher_embed_dim),
            "batter_embedding": torch.randn(pitcher_embed_dim),
            "context": torch.zeros(0),
            "matchup_index": torch.tensor(0),
            "park_index": torch.tensor(0),
            "target": torch.tensor(i % len(OUTCOME_VOCAB)),
        }
        for i in range(n_rows)
    ]

    def collate(batch):
        return {
            "pitcher_embedding": torch.stack([s["pitcher_embedding"] for s in batch]),
            "batter_embedding": torch.stack([s["batter_embedding"] for s in batch]),
            "context": torch.stack([s["context"] for s in batch]),
            "matchup_index": torch.stack([s["matchup_index"] for s in batch]),
            "park_index": torch.stack([s["park_index"] for s in batch]),
            "target": torch.stack([s["target"] for s in batch]),
        }

    loader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=collate)
    predicted_avg_prob, observed_frequency, n = compute_marginal_calibration(model, loader, torch.device("cpu"))

    assert n == n_rows
    assert set(predicted_avg_prob) == set(OUTCOME_VOCAB)
    assert sum(predicted_avg_prob.values()) == pytest.approx(1.0, abs=1e-6)
    assert sum(observed_frequency.values()) == pytest.approx(1.0, abs=1e-6)
    # Deterministic construction: every category i in range(len(OUTCOME_VOCAB)) appears at least once
    # (37 rows > len(OUTCOME_VOCAB)), so every observed frequency should be strictly positive.
    assert all(v > 0 for v in observed_frequency.values())

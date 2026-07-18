from dataclasses import asdict

import pandas as pd
import pytest
import torch
import yaml

from src.data.build_features import build_season_pitches_from_frame
from src.data.contact_quality import build_contact_quality_history, load_raw_batted_balls, save_contact_quality_histories
from src.data.event_embedding_cache import precompute_and_cache_embeddings
from src.data.sequence_dataset import OUTCOME_VOCAB
from src.data.statcast_common import TRAIN_SEASON_RANGE, VAL_SEASONS, build_pitch_frame_from_raw, read_partitioned, write_partitioned
from src.models.long_history_encoder import CareerEncoder, CareerEncoderConfig, ChunkEncoder, ChunkEncoderConfig, LongHistoryEncoder
from src.training.train_event_model import EventTrainingConfig, compute_class_weights, compute_loss_and_metrics
from src.training.train_event_model import main as train_main


def _raw_row(pitcher, batter, game_date, at_bat, pitch_num, balls, strikes, outs, on1, on2, on3,
             home_score, away_score, tto, events, description, season, inning_topbot="Top", launch_speed=None, type_flag="S"):
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
        "launch_speed": launch_speed,
        "type": type_flag,  # "X" = a real batted-ball event (see contact_quality.py); "S"/"B" otherwise, irrelevant to that filter
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


def test_compute_class_weights_gives_rarer_classes_larger_weight():
    # class 0: 100 examples, class 1: 10 examples, class 2: 1 example.
    target = torch.cat([torch.zeros(100, dtype=torch.long), torch.ones(10, dtype=torch.long), torch.full((1,), 2, dtype=torch.long)])
    weights = compute_class_weights(target, num_classes=3, max_weight_ratio=1000.0)
    assert weights[2] > weights[1] > weights[0]


def test_compute_class_weights_normalizes_to_mean_one_across_all_classes():
    target = torch.cat([torch.zeros(100, dtype=torch.long), torch.ones(10, dtype=torch.long)])
    weights = compute_class_weights(target, num_classes=5, max_weight_ratio=1000.0)  # 3 classes entirely absent
    assert weights.mean().item() == pytest.approx(1.0)


def test_compute_class_weights_caps_the_ratio_between_largest_and_smallest_weight():
    target = torch.cat([torch.zeros(1_000_000, dtype=torch.long), torch.ones(1, dtype=torch.long)])
    weights = compute_class_weights(target, num_classes=2, max_weight_ratio=20.0)
    assert (weights.max() / weights.min()).item() == pytest.approx(20.0, rel=1e-3)


def test_compute_class_weights_gives_an_absent_class_a_large_but_finite_weight():
    target = torch.cat([torch.zeros(1000, dtype=torch.long), torch.ones(50, dtype=torch.long)])  # class 2 never appears
    weights = compute_class_weights(target, num_classes=3, max_weight_ratio=1000.0)
    assert torch.isfinite(weights[2])
    assert weights[2] > weights[1] > weights[0]  # absent -> rarer-than-rarest, still respects the overall ordering


def test_compute_loss_and_metrics_returns_none_aux_loss_when_not_given():
    logits = torch.randn(5, len(OUTCOME_VOCAB))
    target = torch.randint(0, len(OUTCOME_VOCAB), (5,))
    main_loss, aux_loss, metrics = compute_loss_and_metrics(logits, target)
    assert aux_loss is None
    assert metrics["aux_loss"] == 0.0


def test_compute_loss_and_metrics_computes_real_mse_when_aux_given():
    logits = torch.randn(5, len(OUTCOME_VOCAB))
    target = torch.randint(0, len(OUTCOME_VOCAB), (5,))
    aux_predictions = torch.tensor([[0.3, 0.4]] * 5)
    aux_targets = torch.tensor([[0.5, 0.4]] * 5)  # column 0 off by 0.2, column 1 exact
    main_loss, aux_loss, metrics = compute_loss_and_metrics(logits, target, aux_predictions=aux_predictions, aux_targets=aux_targets)
    expected_mse = ((0.3 - 0.5) ** 2 + 0.0) / 2  # mean over both columns
    assert aux_loss is not None
    assert aux_loss.item() == pytest.approx(expected_mse)
    assert metrics["aux_loss"] == pytest.approx(expected_mse)


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

    contact_quality_checkpoint = tmp_path / "contact_quality.pkl"
    _write_contact_quality_checkpoint(raw_dir, contact_quality_checkpoint)

    training_config_path = tmp_path / "training_config.yaml"
    _write_training_config(training_config_path)

    log_dir = tmp_path / "logs"
    checkpoint_dir = tmp_path / "checkpoints"

    train_main(
        [
            "--training-config", str(training_config_path),
            "--pitches-dir", str(pitches_dir),
            "--embedding-cache-dir", str(cache_dir),
            "--contact-quality-checkpoint", str(contact_quality_checkpoint),
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
    assert lines[0] == "epoch,train_loss,train_accuracy,train_aux_loss,val_loss,val_accuracy,val_aux_loss"
    assert len(lines) == 3  # header + 2 epochs

    checkpoint_path = checkpoint_dir / "event_model_full_best.pt"
    assert checkpoint_path.exists()
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    assert set(checkpoint.keys()) >= {
        "model_state_dict", "model_config", "park_factor_config", "situational_stats", "contact_quality_stats",
        "epoch", "val_loss", "val_accuracy", "val_aux_loss", "training_metadata",
    }
    assert checkpoint["model_config"]["include_context"] is True

    # Regression test for a real incident (2026-07-17): a full-weight
    # aux-loss checkpoint silently ended up at the default
    # checkpoints/event_model_full_best.pt path with no way to tell it
    # apart from a raw-scalar/unweighted-loss checkpoint short of manually
    # comparing val_loss/epoch against memory. training_metadata exists so
    # that class of mixup is visible from the checkpoint file itself.
    metadata = checkpoint["training_metadata"]
    assert metadata["aux_loss_weight"] == 0.1  # CLI default
    assert metadata["seed"] == 0  # CLI default
    assert metadata["class_weighted_loss"] is False
    assert metadata["include_context"] is True
    assert "saved_at" in metadata


def test_main_train_season_and_val_seasons_flags_override_the_default_split(tmp_path, caplog):
    """Walk-forward retraining needs a non-default season boundary (e.g.
    train through 2023, val 2024) -- confirms --train-season-start/
    --train-season-end/--val-seasons actually change what gets included,
    not just that they parse."""
    raw_dir = tmp_path / "raw"
    pitches_dir = tmp_path / "pitches"
    _write_fixture(raw_dir, pitches_dir)  # one game per season 2015-2023

    cache_dir = tmp_path / "embedding_cache"
    _write_embedding_cache(read_partitioned(pitches_dir), cache_dir)

    contact_quality_checkpoint = tmp_path / "contact_quality.pkl"
    _write_contact_quality_checkpoint(raw_dir, contact_quality_checkpoint)

    training_config_path = tmp_path / "training_config.yaml"
    _write_training_config(training_config_path)

    with caplog.at_level("INFO"):
        train_main(
            [
                "--training-config", str(training_config_path),
                "--pitches-dir", str(pitches_dir),
                "--embedding-cache-dir", str(cache_dir),
                "--contact-quality-checkpoint", str(contact_quality_checkpoint),
                "--train-season-start", "2015",
                "--train-season-end", "2015",
                "--val-seasons", "2016",
                "--epochs", "1",
                "--batch-size", "4",
                "--log-dir", str(tmp_path / "logs"),
                "--checkpoint-dir", str(tmp_path / "checkpoints"),
                "--device", "cpu",
            ]
        )

    assert "Season split -- train: 2015-2015, val: (2016,)" in caplog.text
    checkpoint = torch.load(tmp_path / "checkpoints" / "event_model_full_best.pt", weights_only=False)
    assert checkpoint["epoch"] == 1  # ran successfully on the narrower, overridden split


def test_main_training_metadata_reflects_non_default_aux_loss_weight_and_seed(tmp_path):
    raw_dir = tmp_path / "raw"
    pitches_dir = tmp_path / "pitches"
    _write_fixture(raw_dir, pitches_dir)

    cache_dir = tmp_path / "embedding_cache"
    _write_embedding_cache(read_partitioned(pitches_dir), cache_dir)

    contact_quality_checkpoint = tmp_path / "contact_quality.pkl"
    _write_contact_quality_checkpoint(raw_dir, contact_quality_checkpoint)

    training_config_path = tmp_path / "training_config.yaml"
    _write_training_config(training_config_path)

    train_main(
        [
            "--training-config", str(training_config_path),
            "--pitches-dir", str(pitches_dir),
            "--embedding-cache-dir", str(cache_dir),
            "--contact-quality-checkpoint", str(contact_quality_checkpoint),
            "--epochs", "1",
            "--batch-size", "4",
            "--log-dir", str(tmp_path / "logs"),
            "--checkpoint-dir", str(tmp_path / "checkpoints"),
            "--device", "cpu",
            "--aux-loss-weight", "0.025",
            "--seed", "2",
            "--class-weighted-loss",
        ]
    )

    checkpoint = torch.load(tmp_path / "checkpoints" / "event_model_full_best.pt", weights_only=False)
    metadata = checkpoint["training_metadata"]
    assert metadata["aux_loss_weight"] == 0.025
    assert metadata["seed"] == 2
    assert metadata["class_weighted_loss"] is True


def test_main_interaction_type_flag_reaches_model_config_and_metadata(tmp_path):
    raw_dir = tmp_path / "raw"
    pitches_dir = tmp_path / "pitches"
    _write_fixture(raw_dir, pitches_dir)

    cache_dir = tmp_path / "embedding_cache"
    _write_embedding_cache(read_partitioned(pitches_dir), cache_dir)

    contact_quality_checkpoint = tmp_path / "contact_quality.pkl"
    _write_contact_quality_checkpoint(raw_dir, contact_quality_checkpoint)

    training_config_path = tmp_path / "training_config.yaml"
    _write_training_config(training_config_path)

    train_main(
        [
            "--training-config", str(training_config_path),
            "--pitches-dir", str(pitches_dir),
            "--embedding-cache-dir", str(cache_dir),
            "--contact-quality-checkpoint", str(contact_quality_checkpoint),
            "--epochs", "1",
            "--batch-size", "4",
            "--log-dir", str(tmp_path / "logs"),
            "--checkpoint-dir", str(tmp_path / "checkpoints"),
            "--device", "cpu",
            "--interaction-type", "bilinear",
            "--interaction-dim", "6",
        ]
    )

    checkpoint = torch.load(tmp_path / "checkpoints" / "event_model_full_best.pt", weights_only=False)
    assert checkpoint["model_config"]["interaction_type"] == "bilinear"
    assert checkpoint["model_config"]["interaction_dim"] == 6
    assert checkpoint["training_metadata"]["interaction_type"] == "bilinear"
    assert checkpoint["training_metadata"]["interaction_dim"] == 6


def test_main_defaults_to_unweighted_loss_and_class_weighted_loss_flag_opts_in(tmp_path, caplog):
    raw_dir = tmp_path / "raw"
    pitches_dir = tmp_path / "pitches"
    _write_fixture(raw_dir, pitches_dir)

    cache_dir = tmp_path / "embedding_cache"
    _write_embedding_cache(read_partitioned(pitches_dir), cache_dir)

    contact_quality_checkpoint = tmp_path / "contact_quality.pkl"
    _write_contact_quality_checkpoint(raw_dir, contact_quality_checkpoint)

    training_config_path = tmp_path / "training_config.yaml"
    _write_training_config(training_config_path)

    common_args = [
        "--training-config", str(training_config_path),
        "--pitches-dir", str(pitches_dir),
        "--embedding-cache-dir", str(cache_dir),
        "--contact-quality-checkpoint", str(contact_quality_checkpoint),
        "--epochs", "1",
        "--batch-size", "4",
        "--device", "cpu",
    ]

    with caplog.at_level("INFO"):
        train_main([*common_args, "--log-dir", str(tmp_path / "logs_default"), "--checkpoint-dir", str(tmp_path / "ckpt_default")])
    assert "plain unweighted cross-entropy" in caplog.text
    assert "Class weights" not in caplog.text

    caplog.clear()
    with caplog.at_level("INFO"):
        train_main(
            [*common_args, "--class-weighted-loss", "--log-dir", str(tmp_path / "logs_weighted"), "--checkpoint-dir", str(tmp_path / "ckpt_weighted")]
        )
    assert "Class weights (OUTCOME_VOCAB order)" in caplog.text


def test_main_resumes_from_an_existing_checkpoint_instead_of_restarting(tmp_path):
    raw_dir = tmp_path / "raw"
    pitches_dir = tmp_path / "pitches"
    _write_fixture(raw_dir, pitches_dir)

    cache_dir = tmp_path / "embedding_cache"
    _write_embedding_cache(read_partitioned(pitches_dir), cache_dir)

    contact_quality_checkpoint = tmp_path / "contact_quality.pkl"
    _write_contact_quality_checkpoint(raw_dir, contact_quality_checkpoint)

    training_config_path = tmp_path / "training_config.yaml"
    _write_training_config(training_config_path)

    log_dir = tmp_path / "logs"
    checkpoint_dir = tmp_path / "checkpoints"
    common_args = [
        "--training-config", str(training_config_path),
        "--pitches-dir", str(pitches_dir),
        "--embedding-cache-dir", str(cache_dir),
        "--contact-quality-checkpoint", str(contact_quality_checkpoint),
        "--batch-size", "4",
        "--log-dir", str(log_dir),
        "--checkpoint-dir", str(checkpoint_dir),
        "--device", "cpu",
    ]

    train_main([*common_args, "--epochs", "1"])
    checkpoint_path = checkpoint_dir / "event_model_full_best.pt"
    first_checkpoint = torch.load(checkpoint_path, weights_only=False)
    assert first_checkpoint["epoch"] == 1
    assert "optimizer_state_dict" in first_checkpoint

    log_path = log_dir / "train_event_model_full.csv"
    assert len(log_path.read_text().strip().splitlines()) == 2  # header + epoch 1

    # Rerunning with a higher --epochs (same checkpoint/log dirs) must
    # CONTINUE from epoch 2, not restart training from epoch 1.
    train_main([*common_args, "--epochs", "3"])
    lines = log_path.read_text().strip().splitlines()
    epochs_logged = [int(line.split(",")[0]) for line in lines[1:]]
    assert epochs_logged == [1, 2, 3]  # not [1, 1, 2, 3] -- epoch 1 wasn't redone

    final_checkpoint = torch.load(checkpoint_path, weights_only=False)
    assert final_checkpoint["epoch"] >= first_checkpoint["epoch"]


def test_main_ignores_a_stale_checkpoint_from_a_differently_configured_run_instead_of_crashing(tmp_path):
    """Regression test for a real bug hit during this feature's actual
    retraining run: checkpoint_path is a fixed filename shared across every
    invocation of this script, so a checkpoint left over from a run with a
    different architecture (e.g. before this session's contact-quality
    context feature existed, a smaller situational_dim) can genuinely be
    sitting at that path. Resuming into it must not raise a state_dict
    shape-mismatch RuntimeError -- it should detect the mismatch and start
    fresh instead, the same outcome a pre-resume version of this script
    would have had."""
    raw_dir = tmp_path / "raw"
    pitches_dir = tmp_path / "pitches"
    _write_fixture(raw_dir, pitches_dir)

    cache_dir = tmp_path / "embedding_cache"
    _write_embedding_cache(read_partitioned(pitches_dir), cache_dir)

    contact_quality_checkpoint = tmp_path / "contact_quality.pkl"
    _write_contact_quality_checkpoint(raw_dir, contact_quality_checkpoint)

    training_config_path = tmp_path / "training_config.yaml"
    _write_training_config(training_config_path)

    log_dir = tmp_path / "logs"
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir(parents=True)

    # A checkpoint from a differently-configured run (bogus model_config
    # dict, e.g. a stale situational_dim) sitting at the exact path this
    # run will write to.
    stale_checkpoint_path = checkpoint_dir / "event_model_full_best.pt"
    torch.save(
        {
            "model_state_dict": {},
            "model_config": {"situational_dim": 999, "bogus": True},
            "epoch": 7,
            "val_loss": 0.01,
            "val_accuracy": 0.99,
        },
        stale_checkpoint_path,
    )

    train_main(
        [
            "--training-config", str(training_config_path),
            "--pitches-dir", str(pitches_dir),
            "--embedding-cache-dir", str(cache_dir),
            "--contact-quality-checkpoint", str(contact_quality_checkpoint),
            "--batch-size", "4",
            "--log-dir", str(log_dir),
            "--checkpoint-dir", str(checkpoint_dir),
            "--device", "cpu",
            "--epochs", "1",
        ]
    )

    checkpoint = torch.load(stale_checkpoint_path, weights_only=False)
    assert checkpoint["epoch"] == 1  # started fresh at epoch 1, not "resumed" from the bogus epoch 7
    assert checkpoint["model_config"]["situational_dim"] != 999


def test_main_ignores_a_checkpoint_with_matching_config_but_mismatched_state_dict_keys(tmp_path):
    """Regression test for a real bug this session's actual retraining run
    would have hit: EventModelConfig doesn't have a field for every
    architectural choice (contact_quality_aux_head's existence is implied
    by include_context, not its own config field), so two runs can share an
    identical model_config while their state_dicts have different key
    sets -- a config-only staleness check would call this "resumable" and
    then crash inside load_state_dict on a real key mismatch."""
    raw_dir = tmp_path / "raw"
    pitches_dir = tmp_path / "pitches"
    _write_fixture(raw_dir, pitches_dir)

    cache_dir = tmp_path / "embedding_cache"
    _write_embedding_cache(read_partitioned(pitches_dir), cache_dir)

    contact_quality_checkpoint = tmp_path / "contact_quality.pkl"
    _write_contact_quality_checkpoint(raw_dir, contact_quality_checkpoint)

    training_config_path = tmp_path / "training_config.yaml"
    _write_training_config(training_config_path)

    log_dir = tmp_path / "logs"
    checkpoint_dir = tmp_path / "checkpoints"
    common_args = [
        "--training-config", str(training_config_path),
        "--pitches-dir", str(pitches_dir),
        "--embedding-cache-dir", str(cache_dir),
        "--contact-quality-checkpoint", str(contact_quality_checkpoint),
        "--batch-size", "4",
        "--log-dir", str(log_dir),
        "--checkpoint-dir", str(checkpoint_dir),
        "--device", "cpu",
    ]

    # A real checkpoint from a genuine run of this exact script -- so its
    # model_config WILL match a fresh run's -- but with one key deleted
    # from its state_dict, simulating "an older architecture that happened
    # to produce the same config dict."
    train_main([*common_args, "--epochs", "1"])
    checkpoint_path = checkpoint_dir / "event_model_full_best.pt"
    tampered = torch.load(checkpoint_path, weights_only=False)
    del tampered["model_state_dict"]["contact_quality_aux_head.weight"]
    tampered["epoch"] = 7  # a bogus epoch that would be wrongly "resumed" from if this check failed
    torch.save(tampered, checkpoint_path)

    train_main([*common_args, "--epochs", "1"])  # must not raise

    result = torch.load(checkpoint_path, weights_only=False)
    assert result["epoch"] == 1  # started fresh, not resumed from the tampered epoch 7
    assert "contact_quality_aux_head.weight" in result["model_state_dict"]


def test_main_without_save_as_default_never_overwrites_the_shared_default_checkpoint(tmp_path, monkeypatch):
    """Regression test for the actual 2026-07 incident: an aux-loss-weight
    run had an architecture incompatible with the checkpoint already sitting
    at the shared default path, the staleness check correctly refused to
    resume from it, but nothing then stopped that run from training from
    fresh init and still overwriting the shared default path with its own
    result. A run left at the default --checkpoint-dir without
    --save-as-default must be redirected to checkpoints/experimental/
    instead -- structurally, not just logged after the fact -- so the
    shared/canonical checkpoint can never be silently overwritten this way."""
    raw_dir = tmp_path / "raw"
    pitches_dir = tmp_path / "pitches"
    _write_fixture(raw_dir, pitches_dir)

    cache_dir = tmp_path / "embedding_cache"
    _write_embedding_cache(read_partitioned(pitches_dir), cache_dir)

    contact_quality_checkpoint = tmp_path / "contact_quality.pkl"
    _write_contact_quality_checkpoint(raw_dir, contact_quality_checkpoint)

    training_config_path = tmp_path / "training_config.yaml"
    _write_training_config(training_config_path)

    log_dir = tmp_path / "logs"
    # DEFAULT_CHECKPOINT_DIR ("checkpoints") is a relative path -- chdir into
    # tmp_path so leaving --checkpoint-dir unset in this test can never touch
    # this real repo's actual checkpoints/ directory.
    monkeypatch.chdir(tmp_path)

    common_args = [
        "--training-config", str(training_config_path),
        "--pitches-dir", str(pitches_dir),
        "--embedding-cache-dir", str(cache_dir),
        "--contact-quality-checkpoint", str(contact_quality_checkpoint),
        "--batch-size", "4",
        "--log-dir", str(log_dir),
        "--device", "cpu",
        # deliberately no --checkpoint-dir: this is the "left at the default" case.
    ]

    # A genuine earlier "keeper" run, explicitly opted in via --save-as-default.
    train_main([*common_args, "--epochs", "1", "--save-as-default"])
    default_checkpoint_path = tmp_path / "checkpoints" / "event_model_full_best.pt"
    assert default_checkpoint_path.exists()

    # Tamper it into an incompatible architecture, same technique as the
    # mismatched-state-dict-keys test above -- simulates "a differently
    # configured run's checkpoint is sitting at the shared default path."
    tampered = torch.load(default_checkpoint_path, weights_only=False)
    del tampered["model_state_dict"]["contact_quality_aux_head.weight"]
    tampered["epoch"] = 7  # bogus epoch that would prove a wrongful resume/overwrite
    torch.save(tampered, default_checkpoint_path)

    # A follow-up run that forgot --save-as-default: architecture won't match
    # (fresh init), and without the guard this run would go on to overwrite
    # the shared default path with its own new best checkpoint.
    train_main([*common_args, "--epochs", "1"])

    # The canonical path must be completely untouched.
    still_at_default = torch.load(default_checkpoint_path, weights_only=False)
    assert still_at_default["epoch"] == 7
    assert "contact_quality_aux_head.weight" not in still_at_default["model_state_dict"]

    # The second run's real result must have landed under experimental/ instead.
    experimental_checkpoint_path = tmp_path / "checkpoints" / "experimental" / "event_model_full_best.pt"
    assert experimental_checkpoint_path.exists()
    redirected = torch.load(experimental_checkpoint_path, weights_only=False)
    assert redirected["epoch"] == 1
    assert "contact_quality_aux_head.weight" in redirected["model_state_dict"]  # fresh init has the full architecture


def test_main_with_save_as_default_and_default_checkpoint_dir_writes_the_canonical_path(tmp_path, monkeypatch):
    """Positive-path counterpart: --save-as-default must still let a run
    intentionally write the shared default checkpoint path, so the guard
    added above doesn't block the legitimate "update the keeper checkpoint"
    workflow -- only the accidental/unflagged case."""
    raw_dir = tmp_path / "raw"
    pitches_dir = tmp_path / "pitches"
    _write_fixture(raw_dir, pitches_dir)

    cache_dir = tmp_path / "embedding_cache"
    _write_embedding_cache(read_partitioned(pitches_dir), cache_dir)

    contact_quality_checkpoint = tmp_path / "contact_quality.pkl"
    _write_contact_quality_checkpoint(raw_dir, contact_quality_checkpoint)

    training_config_path = tmp_path / "training_config.yaml"
    _write_training_config(training_config_path)

    log_dir = tmp_path / "logs"
    monkeypatch.chdir(tmp_path)

    train_main([
        "--training-config", str(training_config_path),
        "--pitches-dir", str(pitches_dir),
        "--embedding-cache-dir", str(cache_dir),
        "--contact-quality-checkpoint", str(contact_quality_checkpoint),
        "--batch-size", "4",
        "--log-dir", str(log_dir),
        "--device", "cpu",
        "--epochs", "1",
        "--save-as-default",
    ])

    default_checkpoint_path = tmp_path / "checkpoints" / "event_model_full_best.pt"
    assert default_checkpoint_path.exists()
    experimental_checkpoint_path = tmp_path / "checkpoints" / "experimental" / "event_model_full_best.pt"
    assert not experimental_checkpoint_path.exists()


def test_main_resume_with_epochs_already_satisfied_does_nothing_further(tmp_path):
    raw_dir = tmp_path / "raw"
    pitches_dir = tmp_path / "pitches"
    _write_fixture(raw_dir, pitches_dir)

    cache_dir = tmp_path / "embedding_cache"
    _write_embedding_cache(read_partitioned(pitches_dir), cache_dir)

    contact_quality_checkpoint = tmp_path / "contact_quality.pkl"
    _write_contact_quality_checkpoint(raw_dir, contact_quality_checkpoint)

    training_config_path = tmp_path / "training_config.yaml"
    _write_training_config(training_config_path)

    log_dir = tmp_path / "logs"
    checkpoint_dir = tmp_path / "checkpoints"
    common_args = [
        "--training-config", str(training_config_path),
        "--pitches-dir", str(pitches_dir),
        "--embedding-cache-dir", str(cache_dir),
        "--contact-quality-checkpoint", str(contact_quality_checkpoint),
        "--batch-size", "4",
        "--log-dir", str(log_dir),
        "--checkpoint-dir", str(checkpoint_dir),
        "--device", "cpu",
    ]

    train_main([*common_args, "--epochs", "2"])
    log_path = log_dir / "train_event_model_full.csv"
    lines_after_first_run = log_path.read_text().strip().splitlines()

    # --epochs=2 again, with a checkpoint already at epoch 2 -- no more work,
    # the log must not gain any new rows.
    train_main([*common_args, "--epochs", "2"])
    lines_after_second_run = log_path.read_text().strip().splitlines()
    assert lines_after_second_run == lines_after_first_run


def test_main_with_no_context_writes_a_separate_log_and_checkpoint(tmp_path):
    raw_dir = tmp_path / "raw"
    pitches_dir = tmp_path / "pitches"
    _write_fixture(raw_dir, pitches_dir)

    cache_dir = tmp_path / "embedding_cache"
    _write_embedding_cache(read_partitioned(pitches_dir), cache_dir)

    contact_quality_checkpoint = tmp_path / "contact_quality.pkl"
    _write_contact_quality_checkpoint(raw_dir, contact_quality_checkpoint)

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
            "--contact-quality-checkpoint", str(contact_quality_checkpoint),
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


def test_main_same_seed_produces_identical_initial_weights_different_seed_does_not(tmp_path):
    raw_dir = tmp_path / "raw"
    pitches_dir = tmp_path / "pitches"
    _write_fixture(raw_dir, pitches_dir)

    cache_dir = tmp_path / "embedding_cache"
    _write_embedding_cache(read_partitioned(pitches_dir), cache_dir)

    contact_quality_checkpoint = tmp_path / "contact_quality.pkl"
    _write_contact_quality_checkpoint(raw_dir, contact_quality_checkpoint)

    training_config_path = tmp_path / "training_config.yaml"
    _write_training_config(training_config_path)

    common_args = [
        "--training-config", str(training_config_path),
        "--pitches-dir", str(pitches_dir),
        "--embedding-cache-dir", str(cache_dir),
        "--contact-quality-checkpoint", str(contact_quality_checkpoint),
        "--epochs", "1",
        "--batch-size", "4",
        "--device", "cpu",
    ]

    train_main([*common_args, "--seed", "1", "--log-dir", str(tmp_path / "logs_a"), "--checkpoint-dir", str(tmp_path / "ckpt_a")])
    train_main([*common_args, "--seed", "1", "--log-dir", str(tmp_path / "logs_b"), "--checkpoint-dir", str(tmp_path / "ckpt_b")])
    train_main([*common_args, "--seed", "2", "--log-dir", str(tmp_path / "logs_c"), "--checkpoint-dir", str(tmp_path / "ckpt_c")])

    ckpt_a = torch.load(tmp_path / "ckpt_a" / "event_model_full_best.pt", weights_only=False)
    ckpt_b = torch.load(tmp_path / "ckpt_b" / "event_model_full_best.pt", weights_only=False)
    ckpt_c = torch.load(tmp_path / "ckpt_c" / "event_model_full_best.pt", weights_only=False)

    weight_key = next(iter(ckpt_a["model_state_dict"]))
    assert torch.equal(ckpt_a["model_state_dict"][weight_key], ckpt_b["model_state_dict"][weight_key])
    assert not torch.equal(ckpt_a["model_state_dict"][weight_key], ckpt_c["model_state_dict"][weight_key])

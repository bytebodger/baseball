import numpy as np
import pytest
import yaml
import torch
import pandas as pd

import src.training.pretrain_long_history_encoder as pretrain_long_history_module
from src.data.sequence_dataset import OUTCOME_INDEX, OUTCOME_VOCAB, PlayerPitchSequenceDataset
from src.data.statcast_common import build_pitch_frame_from_raw, write_partitioned
from src.data.build_features import build_season_pitches_from_frame
from src.models.long_history_encoder import CareerEncoder, CareerEncoderConfig, ChunkEncoder, ChunkEncoderConfig
from src.training.pretrain_long_history_encoder import (
    BucketByChunkCountSampler,
    NextPitchLongHistoryDataset,
    NextPitchLongHistoryPredictor,
    _sample_by_pitcher,
    collate_long_history_batch,
    main as pretrain_main,
)


def _raw_row(pitcher, batter, game_pk, game_date, at_bat_number, pitch_number, events=None, description="ball", season=2024):
    return {
        "pitcher": pitcher,
        "batter": batter,
        "game_date": game_date,
        "game_pk": game_pk,
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


def _pitches_frame() -> pd.DataFrame:
    """Pitcher 100: two games in January 2024 (01-01, 3 pitches; 01-06, 4
    pitches) that merge into ONE calendar-month chunk (7 pitches), then one
    game in February 2024 (02-11, 5 pitches) as a second, later chunk --
    exercises both "two games in the same month merge into one chunk" and
    "a new month starts a new chunk even without a huge gap." Pitcher 200:
    a single pitch, single game (career cold start)."""
    rows = []
    for at_bat in range(1, 4):
        rows.append(_raw_row(100, 1, 101, "2024-01-01", at_bat, 1))
    for at_bat in range(1, 5):
        rows.append(_raw_row(100, 1, 102, "2024-01-06", at_bat, 1))
    for at_bat in range(1, 6):
        rows.append(_raw_row(100, 1, 103, "2024-02-11", at_bat, 1))
    rows.append(_raw_row(200, 2, 201, "2024-04-01", 1, 1))

    raw = pd.DataFrame(rows)
    return build_pitch_frame_from_raw(raw)


def _sorted_pitches(pitches: pd.DataFrame) -> pd.DataFrame:
    return pitches.sort_values(["pitcher_id", "game_date", "at_bat_number", "pitch_number"]).reset_index(drop=True)


def _find_idx(sorted_pitches: pd.DataFrame, pitcher_id, game_pk, at_bat_number) -> int:
    match = sorted_pitches[
        (sorted_pitches["pitcher_id"] == pitcher_id)
        & (sorted_pitches["game_pk"] == game_pk)
        & (sorted_pitches["at_bat_number"] == at_bat_number)
    ]
    return match.index[0]


def test_first_pitch_of_a_pitchers_career_has_zero_history():
    pitches = _pitches_frame()
    stats = PlayerPitchSequenceDataset._compute_continuous_stats(pitches)
    dataset = NextPitchLongHistoryDataset(pitches, max_chunks=36, max_pitch_len=200, continuous_stats=stats)
    sorted_pitches = _sorted_pitches(pitches)
    idx = _find_idx(sorted_pitches, 100, 101, 1)

    sample = dataset[idx]
    assert sample["has_history"] is False
    assert sample["num_chunks"] == 0


def test_mid_chunk_target_gets_a_partial_current_month_chunk():
    pitches = _pitches_frame()
    stats = PlayerPitchSequenceDataset._compute_continuous_stats(pitches)
    dataset = NextPitchLongHistoryDataset(pitches, max_chunks=36, max_pitch_len=200, continuous_stats=stats)
    sorted_pitches = _sorted_pitches(pitches)
    idx = _find_idx(sorted_pitches, 100, 101, 2)  # 2nd pitch of career, same game

    sample = dataset[idx]
    assert sample["has_history"] is True
    assert sample["num_chunks"] == 1
    assert sample["chunks"][0]["length"] == 1  # one prior pitch, same month chunk
    assert sample["chunks"][0]["days_before_cutoff"] == pytest.approx(0.0)


def test_new_game_within_the_same_month_continues_the_current_chunk_not_a_new_one():
    """Game 102 starts on 2024-01-06, still January -- unlike the old
    per-game chunking, this must NOT start a new chunk. The first pitch of
    game 102 should see game 101's 3 pitches as a partial *current* chunk
    (not a completed prior chunk), since both games share one month."""
    pitches = _pitches_frame()
    stats = PlayerPitchSequenceDataset._compute_continuous_stats(pitches)
    dataset = NextPitchLongHistoryDataset(pitches, max_chunks=36, max_pitch_len=200, continuous_stats=stats)
    sorted_pitches = _sorted_pitches(pitches)
    idx = _find_idx(sorted_pitches, 100, 102, 1)  # 1st pitch of game 2

    sample = dataset[idx]
    assert sample["num_chunks"] == 1
    assert sample["chunks"][0]["length"] == 3  # all of game 101, merged into this month's chunk
    assert sample["chunks"][0]["days_before_cutoff"] == pytest.approx(5.0)  # 01-06 minus 01-01


def test_target_in_a_new_month_sees_prior_months_merged_game_plus_current_partial():
    """Target deep in game 103 (February) should see: chunk 0 = all of
    January (games 101+102 merged, 7 pitches), chunk 1 = partial February
    (2 prior pitches, same game). days_before_cutoff for chunk 0 must be
    computed from its LAST row's date (game 102's 01-06), not its first
    (game 101's 01-01) -- 36 days, not 41."""
    pitches = _pitches_frame()
    stats = PlayerPitchSequenceDataset._compute_continuous_stats(pitches)
    dataset = NextPitchLongHistoryDataset(pitches, max_chunks=36, max_pitch_len=200, continuous_stats=stats)
    sorted_pitches = _sorted_pitches(pitches)
    idx = _find_idx(sorted_pitches, 100, 103, 3)  # 3rd pitch of game 103

    sample = dataset[idx]
    assert sample["num_chunks"] == 2
    lengths = [c["length"] for c in sample["chunks"]]
    days = [c["days_before_cutoff"] for c in sample["chunks"]]
    assert lengths == [7, 2]  # January (merged games 101+102), February (2 prior pitches)
    assert days == pytest.approx([36.0, 0.0])  # chronological order: oldest first


def test_target_is_the_pitch_own_outcome():
    pitches = _pitches_frame()
    stats = PlayerPitchSequenceDataset._compute_continuous_stats(pitches)
    dataset = NextPitchLongHistoryDataset(pitches, max_chunks=36, max_pitch_len=200, continuous_stats=stats)
    sorted_pitches = _sorted_pitches(pitches)
    idx = _find_idx(sorted_pitches, 200, 201, 1)
    sample = dataset[idx]

    assert sample["target"] == OUTCOME_INDEX[sorted_pitches.loc[idx, "outcome"]]


def test_max_chunks_truncates_to_the_most_recent_months():
    pitches = _pitches_frame()
    stats = PlayerPitchSequenceDataset._compute_continuous_stats(pitches)
    dataset = NextPitchLongHistoryDataset(pitches, max_chunks=1, max_pitch_len=200, continuous_stats=stats)
    sorted_pitches = _sorted_pitches(pitches)
    idx = _find_idx(sorted_pitches, 100, 103, 3)  # would otherwise see 2 chunks (Jan, Feb partial)

    sample = dataset[idx]
    assert sample["num_chunks"] == 1
    # the completed January chunk is dropped, keeping only February's partial chunk
    assert sample["chunks"][0]["length"] == 2
    assert sample["chunks"][0]["days_before_cutoff"] == pytest.approx(0.0)


def test_max_pitch_len_truncates_within_a_single_chunk_keeping_most_recent():
    pitches = _pitches_frame()
    stats = PlayerPitchSequenceDataset._compute_continuous_stats(pitches)
    dataset = NextPitchLongHistoryDataset(pitches, max_chunks=36, max_pitch_len=2, continuous_stats=stats)
    sorted_pitches = _sorted_pitches(pitches)
    idx = _find_idx(sorted_pitches, 100, 102, 1)  # 1st pitch of game 2 -> January chunk (game 101's 3 pitches) so far

    sample = dataset[idx]
    assert sample["num_chunks"] == 1
    assert sample["chunks"][0]["length"] == 2  # capped from 3 down to max_pitch_len=2

    # kept the most recent 2 of game 101's 3 pitches (at-bats 2 and 3, not 1)
    expected_indices = [_find_idx(sorted_pitches, 100, 101, 2), _find_idx(sorted_pitches, 100, 101, 3)]
    expected_continuous = torch.from_numpy(dataset.continuous[expected_indices])
    assert torch.allclose(sample["chunks"][0]["continuous"], expected_continuous)


def test_num_chunks_per_sample_matches_getitem_for_every_sample():
    """The vectorized num_chunks_per_sample precomputation (used by
    BucketByChunkCountSampler) must agree with __getitem__'s own num_chunks
    for every sample, under both a generous and a truncating max_chunks."""
    pitches = _pitches_frame()
    stats = PlayerPitchSequenceDataset._compute_continuous_stats(pitches)
    for max_chunks in (36, 1):
        dataset = NextPitchLongHistoryDataset(pitches, max_chunks=max_chunks, max_pitch_len=200, continuous_stats=stats)
        for idx in range(len(dataset)):
            assert dataset.num_chunks_per_sample[idx] == dataset[idx]["num_chunks"]


def test_cache_dir_tags_the_month_chunking_scheme():
    """A cache directory built under the old per-game scheme must never be
    silently misread as a valid month-based cache -- the cache_dir name
    itself must carry a scheme tag."""
    pitches = _pitches_frame()
    stats = PlayerPitchSequenceDataset._compute_continuous_stats(pitches)
    dataset = NextPitchLongHistoryDataset(
        pitches, max_chunks=36, max_pitch_len=200, continuous_stats=stats, cache_dir="somewhere"
    )
    assert "by_month" in dataset.cache_dir.name
    assert "36" in dataset.cache_dir.name
    assert "200" in dataset.cache_dir.name


def test_uncached_dataset_recomputes_chunk_ranges_every_access(monkeypatch):
    """Without a cache_dir, __getitem__ has no choice but to re-walk chunk
    boundaries from scratch on every single access -- confirms the
    no-cache-dir case actually behaves the way the module docstring says."""
    pitches = _pitches_frame()
    stats = PlayerPitchSequenceDataset._compute_continuous_stats(pitches)
    dataset = NextPitchLongHistoryDataset(pitches, max_chunks=36, max_pitch_len=200, continuous_stats=stats)
    sorted_pitches = _sorted_pitches(pitches)
    idx = _find_idx(sorted_pitches, 100, 103, 3)

    calls = []
    original = dataset._compute_chunk_ranges

    def counting_compute(i):
        calls.append(i)
        return original(i)

    monkeypatch.setattr(dataset, "_compute_chunk_ranges", counting_compute)

    dataset[idx]
    dataset[idx]
    dataset[idx]

    assert calls == [idx, idx, idx]  # recomputed all three times, no memoization


def test_precompute_and_cache_makes_getitem_match_an_uncached_dataset(tmp_path):
    pitches = _pitches_frame()
    stats = PlayerPitchSequenceDataset._compute_continuous_stats(pitches)

    uncached = NextPitchLongHistoryDataset(pitches, max_chunks=36, max_pitch_len=200, continuous_stats=stats)
    cached = NextPitchLongHistoryDataset(
        pitches, max_chunks=36, max_pitch_len=200, continuous_stats=stats, cache_dir=tmp_path
    )
    computed = cached.precompute_and_cache()
    assert computed == len(cached)  # every sample was a cache miss the first time

    for idx in range(len(pitches)):
        expected = uncached[idx]
        actual = cached[idx]
        assert actual["has_history"] == expected["has_history"]
        assert actual["num_chunks"] == expected["num_chunks"]
        for a, e in zip(actual["chunks"], expected["chunks"]):
            assert a["length"] == e["length"]
            assert a["days_before_cutoff"] == pytest.approx(e["days_before_cutoff"])
            assert torch.allclose(a["continuous"], e["continuous"])


def test_cache_hit_skips_recomputation(monkeypatch, tmp_path):
    pitches = _pitches_frame()
    stats = PlayerPitchSequenceDataset._compute_continuous_stats(pitches)

    warm = NextPitchLongHistoryDataset(
        pitches, max_chunks=36, max_pitch_len=200, continuous_stats=stats, cache_dir=tmp_path
    )
    warm.precompute_and_cache()

    # A fresh dataset instance pointed at the same cache_dir should load the
    # precomputed ranges from disk and never re-walk chunk boundaries.
    reloaded = NextPitchLongHistoryDataset(
        pitches, max_chunks=36, max_pitch_len=200, continuous_stats=stats, cache_dir=tmp_path
    )

    def fail_if_called(i):
        raise AssertionError("_compute_chunk_ranges should not be called on a cache hit")

    monkeypatch.setattr(reloaded, "_compute_chunk_ranges", fail_if_called)

    for idx in range(len(pitches)):
        reloaded[idx]  # would raise via fail_if_called if the cache weren't hit


def test_precompute_and_cache_requires_cache_dir():
    pitches = _pitches_frame()
    stats = PlayerPitchSequenceDataset._compute_continuous_stats(pitches)
    dataset = NextPitchLongHistoryDataset(pitches, max_chunks=36, max_pitch_len=200, continuous_stats=stats)

    with pytest.raises(ValueError, match="cache_dir"):
        dataset.precompute_and_cache()


def test_collate_pads_and_masks_mixed_chunk_and_pitch_counts():
    pitches = _pitches_frame()
    stats = PlayerPitchSequenceDataset._compute_continuous_stats(pitches)
    dataset = NextPitchLongHistoryDataset(pitches, max_chunks=36, max_pitch_len=200, continuous_stats=stats)

    batch = [dataset[i] for i in range(len(dataset))]
    chunk_pitch_sequences, days_before_cutoff, chunk_padding_mask, has_history, targets = collate_long_history_batch(batch)

    max_chunks = max(max((sample["num_chunks"] for sample in batch), default=0), 1)
    max_pitch_len = max(max((c["length"] for sample in batch for c in sample["chunks"]), default=0), 1)
    assert chunk_pitch_sequences["continuous"].shape == (len(batch), max_chunks, max_pitch_len, 4)
    assert days_before_cutoff.shape == (len(batch), max_chunks)
    assert targets.shape == (len(batch),)

    zero_history_positions = [i for i, s in enumerate(batch) if not s["has_history"]]
    for i in zero_history_positions:
        assert chunk_padding_mask[i].all()
        assert not has_history[i]


def test_sample_by_pitcher_keeps_each_sampled_pitchers_full_history_intact():
    pitches = _pitches_frame()  # pitcher 100 (12 rows across 3 games), pitcher 200 (1 row)

    sampled = _sample_by_pitcher(pitches, frac=0.5, seed=0)

    kept_pitchers = sampled["pitcher_id"].unique()
    assert len(kept_pitchers) == 1  # round(2 * 0.5) == 1
    kept = kept_pitchers[0]
    # every row for the kept pitcher survives -- not a partial/fragmented history
    assert len(sampled) == len(pitches[pitches["pitcher_id"] == kept])
    assert (sampled["pitcher_id"] == kept).all()


def test_sample_by_pitcher_is_deterministic_given_a_seed():
    pitches = _pitches_frame()
    first = _sample_by_pitcher(pitches, frac=0.5, seed=42)
    second = _sample_by_pitcher(pitches, frac=0.5, seed=42)
    assert first["pitcher_id"].unique().tolist() == second["pitcher_id"].unique().tolist()


def test_next_pitch_long_history_predictor_output_shape():
    chunk_config = ChunkEncoderConfig(hidden_size=8, num_layers=1, num_heads=2, dropout=0.0, feedforward_dim=16, max_seq_len=5)
    career_config = CareerEncoderConfig(hidden_size=8, num_layers=1, num_heads=2, dropout=0.0, feedforward_dim=16, max_chunks=5)
    model = NextPitchLongHistoryPredictor(chunk_config, career_config)

    pitches = _pitches_frame()
    stats = PlayerPitchSequenceDataset._compute_continuous_stats(pitches)
    dataset = NextPitchLongHistoryDataset(pitches, max_chunks=5, max_pitch_len=5, continuous_stats=stats)
    batch = [dataset[i] for i in range(len(dataset))]
    chunk_pitch_sequences, days_before_cutoff, chunk_padding_mask, has_history, targets = collate_long_history_batch(batch)

    logits = model(chunk_pitch_sequences, days_before_cutoff, chunk_padding_mask, has_history)
    assert logits.shape == (len(batch), len(OUTCOME_VOCAB))


def test_bucket_sampler_batches_are_contiguous_slices_of_the_sorted_order():
    counts = np.array([5, 1, 4, 0, 3, 2])
    sampler = BucketByChunkCountSampler(counts, batch_size=2, shuffle=False)

    batches = list(sampler)
    assert len(sampler) == 3
    assert len(batches) == 3
    # sorted order by count: idx 3 (0), idx 1 (1), idx 5 (2), idx 4 (3), idx 2 (4), idx 0 (5)
    assert batches == [[3, 1], [5, 4], [2, 0]]


def test_bucket_sampler_last_batch_may_be_smaller():
    counts = np.array([3, 1, 2])
    sampler = BucketByChunkCountSampler(counts, batch_size=2, shuffle=False)
    batches = list(sampler)
    assert [len(b) for b in batches] == [2, 1]


def test_bucket_sampler_shuffle_reorders_batches_but_not_their_membership():
    counts = np.arange(20)
    sampler = BucketByChunkCountSampler(counts, batch_size=2, shuffle=True, seed=0)

    epoch_1 = [frozenset(b) for b in sampler]
    epoch_2 = [frozenset(b) for b in sampler]

    # same set of batches both epochs (membership fixed by the sort)...
    assert set(epoch_1) == set(epoch_2)
    # ...but each epoch gets its own draw, so the yielded order can differ.
    assert epoch_1 != epoch_2 or len(epoch_1) <= 1


def test_bucket_sampler_shuffle_false_is_deterministic_across_iterations():
    counts = np.array([3, 1, 2, 0])
    sampler = BucketByChunkCountSampler(counts, batch_size=2, shuffle=False)
    assert list(sampler) == list(sampler)


def _write_fake_processed_dataset(base_dir):
    """Real processed pitches (via the real build_features pipeline), several
    games per season so there's actually something to chunk: 3 games in the
    train season (2015), 2 in the val season (2023)."""
    train_rows = []
    for g in range(3):
        game_pk = 1000 + g
        date = f"2015-04-{g * 5 + 1:02d}"
        for at_bat in range(1, 5):
            train_rows.append(_raw_row(100, 1, game_pk, date, at_bat, 1, season=2015))

    val_rows = []
    for g in range(2):
        game_pk = 2000 + g
        date = f"2023-04-{g * 5 + 1:02d}"
        for at_bat in range(1, 4):
            val_rows.append(_raw_row(100, 1, game_pk, date, at_bat, 1, season=2023))

    raw = pd.DataFrame(train_rows + val_rows)
    pitches = build_season_pitches_from_frame(build_pitch_frame_from_raw(raw))
    write_partitioned(pitches, base_dir)


def test_main_runs_end_to_end_and_writes_log_and_checkpoint(tmp_path):
    pitches_dir = tmp_path / "pitches"
    _write_fake_processed_dataset(pitches_dir)

    chunk_config_path = tmp_path / "chunk_config.yaml"
    chunk_config_path.write_text(
        yaml.dump({"hidden_size": 8, "num_layers": 1, "num_heads": 2, "dropout": 0.0, "feedforward_dim": 16, "max_seq_len": 5})
    )
    career_config_path = tmp_path / "career_config.yaml"
    career_config_path.write_text(
        yaml.dump({"hidden_size": 8, "num_layers": 1, "num_heads": 2, "dropout": 0.0, "feedforward_dim": 16, "max_chunks": 5})
    )

    log_dir = tmp_path / "logs"
    checkpoint_dir = tmp_path / "checkpoints"

    pretrain_main(
        [
            "--chunk-config", str(chunk_config_path),
            "--career-config", str(career_config_path),
            "--pitches-dir", str(pitches_dir),
            "--epochs", "1",
            "--batch-size", "4",
            "--log-dir", str(log_dir),
            "--checkpoint-dir", str(checkpoint_dir),
            "--device", "cpu",
        ]
    )

    log_path = log_dir / "pretrain_long_history_encoder.csv"
    assert log_path.exists()
    log_lines = log_path.read_text().strip().splitlines()
    assert log_lines[0] == "epoch,train_loss,train_accuracy,val_loss,val_accuracy"
    assert len(log_lines) == 2  # header + 1 epoch

    checkpoint_path = checkpoint_dir / "long_history_encoder_best.pt"
    assert checkpoint_path.exists()
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    assert checkpoint["epoch"] == 1
    assert set(checkpoint.keys()) >= {
        "chunk_encoder_state_dict",
        "career_encoder_state_dict",
        "classifier_state_dict",
        "chunk_config",
        "career_config",
        "continuous_stats",
        "val_loss",
        "val_accuracy",
    }

    # the saved sub-encoder weights should load cleanly into fresh encoders built from the same configs
    chunk_config = ChunkEncoderConfig(**checkpoint["chunk_config"])
    career_config = CareerEncoderConfig(**checkpoint["career_config"])
    chunk_encoder = ChunkEncoder(chunk_config)
    chunk_encoder.load_state_dict(checkpoint["chunk_encoder_state_dict"])
    career_encoder = CareerEncoder(career_config)
    career_encoder.load_state_dict(checkpoint["career_encoder_state_dict"])


def test_main_with_cache_dir_warms_the_chunk_range_cache(tmp_path):
    pitches_dir = tmp_path / "pitches"
    _write_fake_processed_dataset(pitches_dir)

    chunk_config_path = tmp_path / "chunk_config.yaml"
    chunk_config_path.write_text(
        yaml.dump({"hidden_size": 8, "num_layers": 1, "num_heads": 2, "dropout": 0.0, "feedforward_dim": 16, "max_seq_len": 5})
    )
    career_config_path = tmp_path / "career_config.yaml"
    career_config_path.write_text(
        yaml.dump({"hidden_size": 8, "num_layers": 1, "num_heads": 2, "dropout": 0.0, "feedforward_dim": 16, "max_chunks": 5})
    )

    cache_dir = tmp_path / "chunk_cache"

    pretrain_main(
        [
            "--chunk-config", str(chunk_config_path),
            "--career-config", str(career_config_path),
            "--pitches-dir", str(pitches_dir),
            "--epochs", "1",
            "--batch-size", "4",
            "--log-dir", str(tmp_path / "logs"),
            "--checkpoint-dir", str(tmp_path / "checkpoints"),
            "--device", "cpu",
            "--cache-dir", str(cache_dir),
        ]
    )

    # one cache subdirectory per split, one file per pitcher within each
    assert (cache_dir / "train").exists()
    assert (cache_dir / "val").exists()
    train_cache_files = list((cache_dir / "train").rglob("*.pt"))
    assert len(train_cache_files) > 0


def test_early_stopping_halts_training_after_patience_epochs_without_improvement(tmp_path, monkeypatch):
    pitches_dir = tmp_path / "pitches"
    _write_fake_processed_dataset(pitches_dir)

    chunk_config_path = tmp_path / "chunk_config.yaml"
    chunk_config_path.write_text(
        yaml.dump({"hidden_size": 8, "num_layers": 1, "num_heads": 2, "dropout": 0.0, "feedforward_dim": 16, "max_seq_len": 5})
    )
    career_config_path = tmp_path / "career_config.yaml"
    career_config_path.write_text(
        yaml.dump({"hidden_size": 8, "num_layers": 1, "num_heads": 2, "dropout": 0.0, "feedforward_dim": 16, "max_chunks": 5})
    )

    # Deterministic val_loss trajectory: improves at epoch 1 and 2, then sits
    # flat for 4 consecutive epochs (3-6) -- with patience=4 that should stop
    # training right after epoch 6, even though --epochs allows up to 20.
    val_losses = iter([1.0, 0.9, 0.95, 0.95, 0.95, 0.95, 0.5, 0.5])

    def fake_run_epoch(model, loader, device, criterion, optimizer=None, scaler=None, use_amp=False, log_every=None):
        if optimizer is not None:
            return 1.0, 0.5
        return next(val_losses), 0.5

    monkeypatch.setattr(pretrain_long_history_module, "run_epoch", fake_run_epoch)

    log_dir = tmp_path / "logs"
    checkpoint_dir = tmp_path / "checkpoints"

    pretrain_main(
        [
            "--chunk-config", str(chunk_config_path),
            "--career-config", str(career_config_path),
            "--pitches-dir", str(pitches_dir),
            "--epochs", "20",
            "--patience", "4",
            "--batch-size", "4",
            "--log-dir", str(log_dir),
            "--checkpoint-dir", str(checkpoint_dir),
            "--device", "cpu",
        ]
    )

    log_lines = (log_dir / "pretrain_long_history_encoder.csv").read_text().strip().splitlines()
    assert len(log_lines) == 1 + 6  # header + 6 epochs, not the full 20

    checkpoint = torch.load(checkpoint_dir / "long_history_encoder_best.pt", weights_only=False)
    assert checkpoint["epoch"] == 2  # best val_loss (0.9) was at epoch 2
    assert checkpoint["val_loss"] == 0.9

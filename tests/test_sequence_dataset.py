import pandas as pd
import pytest
import torch

from src.data.sequence_dataset import (
    MATCHUP_INDEX,
    OUTCOME_INDEX,
    PITCH_TYPE_INDEX,
    FallbackPlayerFeatures,
    PlayerPitchSequenceDataset,
)


def _pitch_row(pitcher_id, batter_id, game_date, pitch_number, release_speed, pitch_type="FF", outcome="ball"):
    return {
        "pitcher_id": pitcher_id,
        "batter_id": batter_id,
        "game_date": pd.Timestamp(game_date),
        "game_pk": 1,
        "at_bat_number": pitch_number,  # one pitch per at-bat is fine for these tests
        "pitch_number": 1,
        "pitch_type": pitch_type,
        "release_speed": release_speed,
        "spin_rate": 2200.0,
        "plate_x": 0.1,
        "plate_z": 2.2,
        "stand": "R",
        "p_throws": "L",
        "outcome": outcome,
    }


def _pitches_frame():
    rows = []
    # Pitcher 100: 10 pitches across 10 days, release_speed = 90 + i, so the
    # most recent pitch (10th) is the fastest -- lets tests check truncation
    # keeps the *recent* end, not just any 5 rows.
    for i in range(10):
        rows.append(_pitch_row(100, 1, f"2024-01-{i + 1:02d}", i, release_speed=90.0 + i))

    # Pitcher 200: a single pitch, but dated exactly on the cutoff used in
    # tests below, so it must be excluded ("strictly before").
    rows.append(_pitch_row(200, 2, "2024-02-01", 0, release_speed=95.0))

    # Pitcher 300: exactly 5 pitches (== max_seq_len used below), so no
    # truncation should occur.
    for i in range(5):
        rows.append(_pitch_row(300, 3, f"2024-03-{i + 1:02d}", i, release_speed=80.0 + i))

    return pd.DataFrame(rows)


def test_player_with_lots_of_history_is_truncated_to_most_recent():
    pitches = _pitches_frame()
    dataset = PlayerPitchSequenceDataset(
        pitches, samples=[(100, "2024-02-01")], max_seq_len=5, perspective="pitcher"
    )
    result = dataset[0]

    assert result["has_history"] is True
    assert result["length"] == 5
    assert result["continuous"].shape == (5, 4)
    assert result["position"].tolist() == [0, 1, 2, 3, 4]

    # release_speed was 90..99 across the 10 pitches; truncation should keep
    # the last 5 (95..99), most recent (99, day 10) last.
    mean, std = dataset.continuous_stats["release_speed"]
    kept_speeds = result["continuous"][:, 0] * std + mean
    assert torch.allclose(kept_speeds, torch.tensor([95.0, 96.0, 97.0, 98.0, 99.0]), atol=1e-4)


def test_player_with_zero_history_returns_empty_sequence():
    pitches = _pitches_frame()
    # Pitcher 200's only pitch is dated exactly on the cutoff, which must be
    # excluded since history is "strictly before" the cutoff.
    dataset = PlayerPitchSequenceDataset(
        pitches, samples=[(200, "2024-02-01")], max_seq_len=5, perspective="pitcher"
    )
    result = dataset[0]

    assert result["has_history"] is False
    assert result["length"] == 0
    assert result["continuous"].shape == (0, 4)
    assert result["pitch_type"].shape == (0,)
    assert result["outcome"].shape == (0,)
    assert result["matchup"].shape == (0,)
    assert result["position"].shape == (0,)


def test_player_with_no_rows_at_all_returns_empty_sequence():
    pitches = _pitches_frame()
    dataset = PlayerPitchSequenceDataset(
        pitches, samples=[(999, "2024-06-01")], max_seq_len=5, perspective="pitcher"
    )
    result = dataset[0]
    assert result["has_history"] is False
    assert result["length"] == 0


def test_player_near_sequence_length_cap_is_not_truncated():
    pitches = _pitches_frame()
    dataset = PlayerPitchSequenceDataset(
        pitches, samples=[(300, "2024-04-01")], max_seq_len=5, perspective="pitcher"
    )
    result = dataset[0]

    assert result["has_history"] is True
    assert result["length"] == 5  # exactly max_seq_len, all of it kept
    mean, std = dataset.continuous_stats["release_speed"]
    kept_speeds = result["continuous"][:, 0] * std + mean
    assert torch.allclose(kept_speeds, torch.tensor([80.0, 81.0, 82.0, 83.0, 84.0]), atol=1e-4)


def test_categorical_features_are_valid_indices():
    pitches = _pitches_frame()
    dataset = PlayerPitchSequenceDataset(
        pitches, samples=[(100, "2024-02-01")], max_seq_len=5, perspective="pitcher"
    )
    result = dataset[0]

    assert (result["pitch_type"] == PITCH_TYPE_INDEX["FF"]).all()
    assert (result["outcome"] == OUTCOME_INDEX["ball"]).all()
    assert (result["matchup"] == MATCHUP_INDEX["R_L"]).all()  # stand=R, p_throws=L


def test_batter_perspective_selects_by_batter_id():
    pitches = _pitches_frame()
    dataset = PlayerPitchSequenceDataset(
        pitches, samples=[(1, "2024-02-01")], max_seq_len=5, perspective="batter"
    )
    result = dataset[0]
    assert result["has_history"] is True
    assert result["length"] == 5  # same 10 pitches belong to batter_id=1 too


def test_fallback_features_computes_age_when_bio_available():
    bio = pd.DataFrame({"player_id": [100], "birth_date": [pd.Timestamp("1994-02-01")]})
    fallback = FallbackPlayerFeatures(player_bio=bio)

    features = fallback.get_features(100, "2024-02-01")
    assert features["age"] == pytest.approx(30.0, abs=0.01)
    assert features["minor_league_stats"] is None


def test_fallback_features_age_is_nan_without_bio():
    fallback = FallbackPlayerFeatures()
    features = fallback.get_features(100, "2024-02-01")
    assert pd.isna(features["age"])


def test_fallback_features_age_is_nan_for_unknown_player():
    bio = pd.DataFrame({"player_id": [100], "birth_date": [pd.Timestamp("1994-02-01")]})
    fallback = FallbackPlayerFeatures(player_bio=bio)
    features = fallback.get_features(999, "2024-02-01")
    assert pd.isna(features["age"])


def test_precompute_and_cache_writes_a_file_readable_by_a_fresh_instance(tmp_path):
    pitches = _pitches_frame()
    cache_dir = tmp_path / "cache"

    warm = PlayerPitchSequenceDataset(pitches, samples=[], max_seq_len=5, perspective="pitcher", cache_dir=cache_dir)
    computed = warm.precompute_and_cache([(100, "2024-02-01")])
    assert computed == 1
    assert (cache_dir / "pitcher" / "100.pt").exists()

    # a brand-new instance (simulating a later epoch, or a separate script
    # run like backtest.py) must read the identical cached result rather
    # than recompute it, and never write anything itself.
    reader = PlayerPitchSequenceDataset(pitches, samples=[], max_seq_len=5, perspective="pitcher", cache_dir=cache_dir)
    from_cache = reader.build_sequence(100, "2024-02-01")
    from_scratch = PlayerPitchSequenceDataset(
        pitches, samples=[], max_seq_len=5, perspective="pitcher"
    ).build_sequence(100, "2024-02-01")

    assert torch.equal(from_cache["continuous"], from_scratch["continuous"])
    assert from_cache["length"] == from_scratch["length"]


def test_precompute_and_cache_skips_already_cached_queries(tmp_path):
    pitches = _pitches_frame()
    cache_dir = tmp_path / "cache"
    ds = PlayerPitchSequenceDataset(pitches, samples=[], max_seq_len=5, perspective="pitcher", cache_dir=cache_dir)

    first_pass = ds.precompute_and_cache([(100, "2024-02-01"), (300, "2024-04-01")])
    second_pass = ds.precompute_and_cache([(100, "2024-02-01"), (300, "2024-04-01")])

    assert first_pass == 2
    assert second_pass == 0  # both already cached, nothing new computed


def test_build_sequence_falls_back_to_computing_on_a_cache_miss(tmp_path):
    """A query never warmed still works (just not from cache) -- build_sequence
    must never require a fully-warmed cache to function correctly."""
    pitches = _pitches_frame()
    cache_dir = tmp_path / "cache"
    ds = PlayerPitchSequenceDataset(pitches, samples=[], max_seq_len=5, perspective="pitcher", cache_dir=cache_dir)

    result = ds.build_sequence(100, "2024-02-01")
    assert result["has_history"] is True
    assert result["length"] == 5
    # build_sequence itself must not have written to disk
    assert not (cache_dir / "pitcher" / "100.pt").exists()


def test_precompute_and_cache_requires_cache_dir():
    pitches = _pitches_frame()
    ds = PlayerPitchSequenceDataset(pitches, samples=[], max_seq_len=5, perspective="pitcher")
    with pytest.raises(ValueError):
        ds.precompute_and_cache([(100, "2024-02-01")])

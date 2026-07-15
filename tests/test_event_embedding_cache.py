import pandas as pd
import pytest
import torch

from src.data.event_embedding_cache import (
    EmbeddingCache,
    QueryChunkedHistoryDataset,
    build_chunk_index,
    chunk_ranges_for_query,
    distinct_pairs,
    estimate_num_chunks,
    precompute_and_cache_embeddings,
)
from src.models.long_history_encoder import CareerEncoder, CareerEncoderConfig, ChunkEncoder, ChunkEncoderConfig, LongHistoryEncoder


def _small_chunk_config() -> ChunkEncoderConfig:
    return ChunkEncoderConfig(hidden_size=8, num_layers=1, num_heads=2, dropout=0.0, feedforward_dim=16, max_seq_len=10)


def _small_career_config(max_chunks: int = 6) -> CareerEncoderConfig:
    return CareerEncoderConfig(hidden_size=8, num_layers=1, num_heads=2, dropout=0.0, feedforward_dim=16, max_chunks=max_chunks)


def _small_encoder() -> LongHistoryEncoder:
    torch.manual_seed(0)
    return LongHistoryEncoder(ChunkEncoder(_small_chunk_config()), CareerEncoder(_small_career_config()))


def _pitches_fixture() -> pd.DataFrame:
    """Pitcher 1 pitches in April and May 2023 (two chunks); pitcher 2 only
    in April. Batter 10 bats against both pitchers across both months;
    batter 11 only appears in May. Enough rows per (player, month) to
    exercise real chunk boundaries."""
    rows = []

    def add(pitcher, batter, date, at_bat, pitch_num, pitch_type, outcome):
        rows.append(
            {
                "pitcher_id": pitcher,
                "batter_id": batter,
                "game_date": pd.Timestamp(date),
                "at_bat_number": at_bat,
                "pitch_number": pitch_num,
                "pitch_type": pitch_type,
                "release_speed": 92.0,
                "spin_rate": 2200.0,
                "plate_x": 0.1,
                "plate_z": 2.3,
                "stand": "R",
                "p_throws": "R",
                "outcome": outcome,
            }
        )

    # April: pitcher 1 vs batter 10.
    add(1, 10, "2023-04-05", 1, 1, "FF", "ball")
    add(1, 10, "2023-04-05", 1, 2, "FF", "called_strike")
    add(1, 10, "2023-04-12", 2, 1, "SL", "strikeout")
    # April: pitcher 2 vs batter 10.
    add(2, 10, "2023-04-20", 1, 1, "CH", "single")
    # May: pitcher 1 vs batter 10 and batter 11.
    add(1, 10, "2023-05-03", 1, 1, "FF", "walk")
    add(1, 11, "2023-05-10", 2, 1, "SL", "home_run")

    return pd.DataFrame(rows)


# ---------- build_chunk_index / chunk_ranges_for_query ----------


def test_chunk_ranges_for_query_only_includes_pitches_strictly_before_cutoff():
    pitches = _pitches_fixture()
    index = build_chunk_index(pitches, "pitcher_id")
    # Querying pitcher 1 as of 2023-05-03 (the May pitch's own date) should
    # only see April's 3 pitches -- May hasn't happened yet as of its own date.
    cutoff_ns = pd.Timestamp("2023-05-03").value
    resolved = chunk_ranges_for_query(index, 1, cutoff_ns, max_chunks=6, max_pitch_len=10)
    total_pitches = sum(end - start for start, end, _ in resolved)
    assert total_pitches == 3


def test_chunk_ranges_for_query_unknown_player_returns_empty():
    pitches = _pitches_fixture()
    index = build_chunk_index(pitches, "pitcher_id")
    resolved = chunk_ranges_for_query(index, 999, pd.Timestamp("2023-06-01").value, max_chunks=6, max_pitch_len=10)
    assert resolved == []


def test_chunk_ranges_for_query_groups_by_calendar_month():
    pitches = _pitches_fixture()
    index = build_chunk_index(pitches, "pitcher_id")
    # As of June, pitcher 1's history spans April (3 pitches: 2 on 04-05, 1
    # on 04-12) and May (2 pitches: 05-03 and 05-10) -- 2 chunks.
    resolved = chunk_ranges_for_query(index, 1, pd.Timestamp("2023-06-01").value, max_chunks=6, max_pitch_len=10)
    assert len(resolved) == 2
    lengths = [end - start for start, end, _ in resolved]
    assert sorted(lengths) == [2, 3]


def test_estimate_num_chunks_matches_len_of_resolved_chunk_ranges():
    pitches = _pitches_fixture()
    index = build_chunk_index(pitches, "pitcher_id")
    queries = [(1, pd.Timestamp("2023-06-01").value), (1, pd.Timestamp("2023-04-06").value), (999, pd.Timestamp("2023-06-01").value)]
    estimated = estimate_num_chunks(index, queries, max_chunks=6)
    actual = [len(chunk_ranges_for_query(index, p, d, max_chunks=6, max_pitch_len=10)) for p, d in queries]
    assert estimated.tolist() == actual


def test_distinct_pairs_deduplicates_same_day_multiple_pitches():
    pitches = _pitches_fixture()
    pairs = distinct_pairs(pitches, "pitcher")
    # Pitcher 1 threw 2 pitches on 2023-04-05 but that's one distinct pair.
    assert (1, pd.Timestamp("2023-04-05").value) in pairs
    assert pairs.count((1, pd.Timestamp("2023-04-05").value)) == 1


# ---------- precompute_and_cache_embeddings / EmbeddingCache ----------


def test_precompute_writes_one_file_per_entry_per_perspective(tmp_path):
    pitches = _pitches_fixture()
    encoder = _small_encoder()
    counts = precompute_and_cache_embeddings(
        pitches, encoder, tmp_path, max_chunks=6, max_pitch_len=10, device=torch.device("cpu"), batch_size=4
    )
    assert counts["pitcher"] == len(distinct_pairs(pitches, "pitcher"))
    assert counts["batter"] == len(distinct_pairs(pitches, "batter"))

    pitcher1_dates = {d for p, d in distinct_pairs(pitches, "pitcher") if p == 1}
    for date_ns in pitcher1_dates:
        entry = tmp_path / "pitcher" / "1" / f"{date_ns}.pt"
        assert entry.exists()
        assert torch.load(entry, weights_only=False).shape == (8,)  # small_career_config's hidden_size

    # No old-format per-player dict file should be created by fresh writes.
    assert not (tmp_path / "pitcher" / "1.pt").exists()
    assert (tmp_path / "batter" / "10").is_dir()
    assert (tmp_path / "batter" / "11").is_dir()


def test_precompute_skips_already_cached_pairs_without_recomputing(tmp_path):
    pitches = _pitches_fixture()
    encoder = _small_encoder()

    # Pre-seed pitcher 1's cache with a poisoned sentinel value for one real
    # date, in the new per-entry format -- if precompute recomputed it, this
    # exact value wouldn't survive.
    poisoned_date = pd.Timestamp("2023-04-05").value
    sentinel = torch.full((8,), 999.0)
    entry_dir = tmp_path / "pitcher" / "1"
    entry_dir.mkdir(parents=True)
    torch.save(sentinel, entry_dir / f"{poisoned_date}.pt")

    counts = precompute_and_cache_embeddings(
        pitches, encoder, tmp_path, max_chunks=6, max_pitch_len=10, device=torch.device("cpu"), batch_size=4
    )
    total_pitcher_pairs = len(distinct_pairs(pitches, "pitcher"))
    # Every pair except the one poisoned/pre-cached one should have been computed.
    assert counts["pitcher"] == total_pitcher_pairs - 1

    assert torch.equal(torch.load(entry_dir / f"{poisoned_date}.pt", weights_only=False), sentinel)
    # The other pitcher-1 dates should now also be present (newly computed).
    expected_dates = {d for p, d in distinct_pairs(pitches, "pitcher") if p == 1}
    assert {int(p.stem) for p in entry_dir.glob("*.pt")} == expected_dates


def test_precompute_skips_pairs_already_cached_under_the_old_per_player_dict_format(tmp_path):
    """The ~96% of pairs cached before the entry-per-file format existed
    don't get migrated -- precompute must still recognize them as done by
    reading (not rewriting) the old dict file, and only fill the gap."""
    pitches = _pitches_fixture()
    encoder = _small_encoder()

    poisoned_date = pd.Timestamp("2023-04-05").value
    sentinel = torch.full((8,), 999.0)
    pitcher_dir = tmp_path / "pitcher"
    pitcher_dir.mkdir(parents=True)
    torch.save({poisoned_date: sentinel}, pitcher_dir / "1.pt")

    counts = precompute_and_cache_embeddings(
        pitches, encoder, tmp_path, max_chunks=6, max_pitch_len=10, device=torch.device("cpu"), batch_size=4
    )
    total_pitcher_pairs = len(distinct_pairs(pitches, "pitcher"))
    assert counts["pitcher"] == total_pitcher_pairs - 1

    # Old-format file is untouched (still just the one poisoned entry).
    reloaded = torch.load(pitcher_dir / "1.pt", weights_only=False)
    assert set(reloaded.keys()) == {poisoned_date}
    assert torch.equal(reloaded[poisoned_date], sentinel)

    # The newly computed dates landed as new-format entry files instead.
    expected_new_dates = {d for p, d in distinct_pairs(pitches, "pitcher") if p == 1} - {poisoned_date}
    assert {int(p.stem) for p in (pitcher_dir / "1").glob("*.pt")} == expected_new_dates


def test_precompute_writes_survive_a_mid_run_crash(tmp_path):
    """Incremental-checkpointing check: force the encoder to raise on its
    second batch and confirm the first batch's entry files already landed on
    disk before the crash -- proving writes aren't deferred to the end of
    the whole perspective's loop (which would lose everything on a kill/OOM/
    hang, as happened in practice -- see git history around this module)."""
    pitches = _pitches_fixture()
    encoder = _small_encoder()

    call_count = {"n": 0}
    original_forward = encoder.forward

    def flaky_forward(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 2:
            raise RuntimeError("simulated crash mid-run")
        return original_forward(*args, **kwargs)

    encoder.forward = flaky_forward

    with pytest.raises(RuntimeError, match="simulated crash"):
        precompute_and_cache_embeddings(
            pitches, encoder, tmp_path, max_chunks=6, max_pitch_len=10, device=torch.device("cpu"), batch_size=1
        )

    # The first (successful) batch's entry file(s) must have survived the
    # crash on the second batch -- at least one entry exists, and the run as
    # a whole did NOT complete (fewer total cached pairs than the full set).
    entry_files = list((tmp_path / "pitcher").glob("*/*.pt")) + list((tmp_path / "batter").glob("*/*.pt"))
    assert len(entry_files) >= 1

    total_cached = len(entry_files)
    total_pairs = len(distinct_pairs(pitches, "pitcher")) + len(distinct_pairs(pitches, "batter"))
    assert 0 < total_cached < total_pairs


def test_precompute_is_idempotent_second_call_computes_nothing_new(tmp_path):
    pitches = _pitches_fixture()
    encoder = _small_encoder()
    precompute_and_cache_embeddings(pitches, encoder, tmp_path, max_chunks=6, max_pitch_len=10, device=torch.device("cpu"), batch_size=4)
    counts = precompute_and_cache_embeddings(
        pitches, encoder, tmp_path, max_chunks=6, max_pitch_len=10, device=torch.device("cpu"), batch_size=4
    )
    assert counts == {"pitcher": 0, "batter": 0}


def test_embedding_cache_get_matches_the_value_precompute_wrote(tmp_path):
    pitches = _pitches_fixture()
    encoder = _small_encoder()
    precompute_and_cache_embeddings(pitches, encoder, tmp_path, max_chunks=6, max_pitch_len=10, device=torch.device("cpu"), batch_size=4)

    cache = EmbeddingCache(tmp_path, "pitcher")
    embedding = cache.get(1, "2023-04-05")
    raw = torch.load(tmp_path / "pitcher" / "1" / f"{pd.Timestamp('2023-04-05').value}.pt", weights_only=False)
    assert torch.equal(embedding, raw)


def test_embedding_cache_get_falls_back_to_old_per_player_dict_format(tmp_path):
    """Entries cached before the entry-per-file format existed are never
    migrated -- EmbeddingCache must still be able to read them."""
    date_ns = pd.Timestamp("2023-04-05").value
    embedding = torch.arange(8, dtype=torch.float32)
    pitcher_dir = tmp_path / "pitcher"
    pitcher_dir.mkdir(parents=True)
    torch.save({date_ns: embedding}, pitcher_dir / "1.pt")

    cache = EmbeddingCache(tmp_path, "pitcher")
    assert torch.equal(cache.get(1, "2023-04-05"), embedding)


def test_embedding_cache_get_prefers_new_format_when_both_exist_for_the_same_player(tmp_path):
    """A player can have an old-format dict file for some dates and new
    per-entry files for others (dates added after the format switch) -- a
    lookup for a new-format date must find it even though the old dict file
    also exists (just without that date)."""
    old_date = pd.Timestamp("2023-04-05").value
    new_date = pd.Timestamp("2023-04-12").value
    old_embedding = torch.zeros(8)
    new_embedding = torch.ones(8)

    pitcher_dir = tmp_path / "pitcher"
    pitcher_dir.mkdir(parents=True)
    torch.save({old_date: old_embedding}, pitcher_dir / "1.pt")
    (pitcher_dir / "1").mkdir()
    torch.save(new_embedding, pitcher_dir / "1" / f"{new_date}.pt")

    cache = EmbeddingCache(tmp_path, "pitcher")
    assert torch.equal(cache.get(1, "2023-04-05"), old_embedding)
    assert torch.equal(cache.get(1, "2023-04-12"), new_embedding)


def test_embedding_cache_get_raises_keyerror_on_miss(tmp_path):
    pitches = _pitches_fixture()
    encoder = _small_encoder()
    precompute_and_cache_embeddings(pitches, encoder, tmp_path, max_chunks=6, max_pitch_len=10, device=torch.device("cpu"), batch_size=4)

    cache = EmbeddingCache(tmp_path, "pitcher")
    with pytest.raises(KeyError):
        cache.get(1, "2099-01-01")  # never-computed date
    with pytest.raises(KeyError):
        cache.get(9999, "2023-04-05")  # never-seen player


def test_embedding_cache_rejects_invalid_perspective(tmp_path):
    with pytest.raises(ValueError):
        EmbeddingCache(tmp_path, "catcher")


def test_embedding_cache_get_batch_matches_individual_get_calls(tmp_path):
    pitches = _pitches_fixture()
    encoder = _small_encoder()
    precompute_and_cache_embeddings(pitches, encoder, tmp_path, max_chunks=6, max_pitch_len=10, device=torch.device("cpu"), batch_size=4)

    cache = EmbeddingCache(tmp_path, "batter")
    player_ids = pd.Series([10, 10, 11])
    dates = pd.Series(["2023-04-05", "2023-05-03", "2023-05-10"])
    batched = cache.get_batch(player_ids, dates)
    individual = torch.stack([cache.get(p, d) for p, d in zip(player_ids, dates)])
    assert torch.equal(batched, individual)


def test_pitcher_and_batter_caches_are_independent_for_the_same_numeric_id(tmp_path):
    # A player_id that's a pitcher in one cache and coincidentally queried as
    # a batter in the other shouldn't collide -- separate subdirectories.
    pitches = _pitches_fixture()
    encoder = _small_encoder()
    precompute_and_cache_embeddings(pitches, encoder, tmp_path, max_chunks=6, max_pitch_len=10, device=torch.device("cpu"), batch_size=4)

    # pitcher_id=1 exists in the pitcher cache but never appears as a batter_id.
    pitcher_cache = EmbeddingCache(tmp_path, "pitcher")
    batter_cache = EmbeddingCache(tmp_path, "batter")
    assert pitcher_cache.get(1, "2023-04-05") is not None
    with pytest.raises(KeyError):
        batter_cache.get(1, "2023-04-05")


def test_query_chunked_history_dataset_no_history_sample_has_history_false():
    pitches = _pitches_fixture()
    index = build_chunk_index(pitches, "pitcher_id")
    # Pitcher 1's very first pitch date: nothing strictly before it.
    dataset = QueryChunkedHistoryDataset(index, [(1, pd.Timestamp("2023-04-05").value)], max_chunks=6, max_pitch_len=10)
    sample = dataset[0]
    assert sample["has_history"] is False
    assert sample["num_chunks"] == 0

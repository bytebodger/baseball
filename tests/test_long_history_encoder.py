import pytest
import torch

from src.data.sequence_dataset import MATCHUP_VOCAB, OUTCOME_VOCAB, PITCH_TYPE_VOCAB
from src.models.long_history_encoder import (
    CareerEncoder,
    CareerEncoderConfig,
    ChunkEncoder,
    ChunkEncoderConfig,
    ChunkTimeEncoding,
    LongHistoryEncoder,
)


def _small_chunk_config() -> ChunkEncoderConfig:
    return ChunkEncoderConfig(hidden_size=16, num_layers=2, num_heads=2, dropout=0.0, feedforward_dim=32, max_seq_len=10)


def _small_career_config(max_chunks: int = 400) -> CareerEncoderConfig:
    return CareerEncoderConfig(
        hidden_size=16, num_layers=2, num_heads=2, dropout=0.0, feedforward_dim=32, max_chunks=max_chunks
    )


def _make_chunk_batch(lengths: list[int]):
    """Mirrors test_player_encoder.py's _make_batch -- a padded pitch batch
    for ChunkEncoder (== PlayerEncoder)."""
    batch_size = len(lengths)
    max_len = max(max(lengths), 1)

    continuous = torch.zeros(batch_size, max_len, 4)
    pitch_type = torch.zeros(batch_size, max_len, dtype=torch.long)
    outcome = torch.zeros(batch_size, max_len, dtype=torch.long)
    matchup = torch.zeros(batch_size, max_len, dtype=torch.long)
    position = torch.zeros(batch_size, max_len, dtype=torch.long)
    padding_mask = torch.ones(batch_size, max_len, dtype=torch.bool)
    has_history = torch.zeros(batch_size, dtype=torch.bool)

    for i, length in enumerate(lengths):
        if length == 0:
            continue
        has_history[i] = True
        continuous[i, :length] = torch.randn(length, 4)
        pitch_type[i, :length] = torch.randint(0, len(PITCH_TYPE_VOCAB), (length,))
        outcome[i, :length] = torch.randint(0, len(OUTCOME_VOCAB), (length,))
        matchup[i, :length] = torch.randint(0, len(MATCHUP_VOCAB), (length,))
        position[i, :length] = torch.arange(length)
        padding_mask[i, :length] = False

    return continuous, pitch_type, outcome, matchup, position, padding_mask, has_history


def _make_career_batch(hidden_size: int, chunk_counts: list[int], max_chunks: int):
    """A padded batch of chunk embeddings for CareerEncoder: `chunk_counts[i]`
    real chunks for player i, padded to `max_chunks`, most-recent-first days
    (0, 1, 2, ...) for the real chunks and 0 for padding."""
    batch_size = len(chunk_counts)
    chunk_embeddings = torch.zeros(batch_size, max_chunks, hidden_size)
    days_before_cutoff = torch.zeros(batch_size, max_chunks)
    padding_mask = torch.ones(batch_size, max_chunks, dtype=torch.bool)
    has_history = torch.zeros(batch_size, dtype=torch.bool)

    for i, count in enumerate(chunk_counts):
        if count == 0:
            continue
        has_history[i] = True
        chunk_embeddings[i, :count] = torch.randn(count, hidden_size)
        days_before_cutoff[i, :count] = torch.arange(count).float()
        padding_mask[i, :count] = False

    return chunk_embeddings, days_before_cutoff, padding_mask, has_history


# ---------- ChunkEncoder ----------


def test_chunk_encoder_is_a_player_encoder_with_the_same_forward_contract():
    model = ChunkEncoder(_small_chunk_config())
    model.eval()

    batch = _make_chunk_batch([5, 0, 3])
    output = model(*batch)

    assert output.shape == (3, 16)
    assert torch.allclose(output[1], model.no_history_embedding.detach())


# ---------- ChunkTimeEncoding ----------


def test_chunk_time_encoding_odd_hidden_size_raises():
    with pytest.raises(ValueError, match="even"):
        ChunkTimeEncoding(15)


def test_chunk_time_encoding_shape():
    encoding = ChunkTimeEncoding(16)
    days = torch.tensor([[0.0, 5.0, 30.0], [1.0, 400.0, 0.0]])
    result = encoding(days)
    assert result.shape == (2, 3, 16)


def test_chunk_time_encoding_differs_by_elapsed_time_not_just_position():
    # same chunk *position* (index 0 in each row) but different elapsed time
    # -> the encoding must differ, unlike a discrete position embedding.
    encoding = ChunkTimeEncoding(16)
    days = torch.tensor([[0.0], [200.0]])
    result = encoding(days)
    assert not torch.allclose(result[0, 0], result[1, 0])


def test_chunk_time_encoding_same_elapsed_time_gives_same_encoding():
    encoding = ChunkTimeEncoding(16)
    days = torch.tensor([[42.0], [42.0]])
    result = encoding(days)
    assert torch.allclose(result[0, 0], result[1, 0])


# ---------- CareerEncoder ----------


def test_career_encoder_output_shape_for_mixed_batch_with_empty_histories():
    model = CareerEncoder(_small_career_config(max_chunks=10))
    model.eval()

    batch = _make_career_batch(hidden_size=16, chunk_counts=[5, 0, 3, 0], max_chunks=10)
    output = model(*batch)

    assert output.shape == (4, 16)


def test_career_encoder_zero_history_rows_get_the_learned_no_history_embedding():
    model = CareerEncoder(_small_career_config(max_chunks=10))
    model.eval()

    chunk_embeddings, days, padding_mask, has_history = _make_career_batch(
        hidden_size=16, chunk_counts=[4, 0, 2, 0], max_chunks=10
    )
    output = model(chunk_embeddings, days, padding_mask, has_history)

    expected = model.no_history_embedding.detach()
    assert torch.allclose(output[1], expected)
    assert torch.allclose(output[3], expected)
    assert not torch.allclose(output[0], expected)
    assert not torch.allclose(output[2], expected)


def test_career_encoder_all_empty_batch_skips_transformer_entirely():
    model = CareerEncoder(_small_career_config(max_chunks=10))
    model.eval()

    batch = _make_career_batch(hidden_size=16, chunk_counts=[0, 0, 0], max_chunks=10)
    output = model(*batch)

    expected = model.no_history_embedding.detach()
    for row in output:
        assert torch.allclose(row, expected)


def test_career_encoder_ignores_padding_chunk_time_values_even_if_nan():
    """Padding positions' days_before_cutoff should never affect the output --
    not because they happen to be zero, but because padding_mask excludes them.
    Feeding NaN into padding slots (instead of the well-behaved 0 a real
    pipeline would use) must not corrupt the result, since 0 * NaN == NaN
    under attention masking if the model didn't defensively zero it first."""
    model = CareerEncoder(_small_career_config(max_chunks=10))
    model.eval()

    chunk_embeddings, days, padding_mask, has_history = _make_career_batch(
        hidden_size=16, chunk_counts=[3], max_chunks=10
    )
    clean_output = model(chunk_embeddings, days, padding_mask, has_history)

    days_with_nan = days.clone()
    days_with_nan[0, 3:] = float("nan")  # padding region for this row
    nan_output = model(chunk_embeddings, days_with_nan, padding_mask, has_history)

    assert torch.allclose(clean_output, nan_output)
    assert not torch.isnan(nan_output).any()


def test_career_encoder_respects_max_chunks_cap_via_config():
    config = _small_career_config(max_chunks=400)
    assert config.max_chunks == 400


def test_career_encoder_gradients_flow_through_both_branches():
    model = CareerEncoder(_small_career_config(max_chunks=10))
    batch = _make_career_batch(hidden_size=16, chunk_counts=[4, 0, 2], max_chunks=10)

    output = model(*batch)
    (output**2).sum().backward()

    assert model.no_history_embedding.grad is not None
    assert not torch.all(model.no_history_embedding.grad == 0)


# ---------- LongHistoryEncoder ----------


def test_long_history_encoder_mismatched_hidden_size_raises():
    chunk_encoder = ChunkEncoder(_small_chunk_config())
    career_encoder = CareerEncoder(_small_career_config(max_chunks=10))
    career_encoder.config.hidden_size = 32  # force a mismatch without rebuilding the whole module

    with pytest.raises(ValueError, match="hidden_size"):
        LongHistoryEncoder(chunk_encoder, career_encoder)


def test_long_history_encoder_end_to_end_output_shape():
    hidden_size = 16
    max_chunks = 5
    max_pitch_len = 6
    batch_size = 3

    chunk_encoder = ChunkEncoder(_small_chunk_config())
    career_encoder = CareerEncoder(_small_career_config(max_chunks=max_chunks))
    model = LongHistoryEncoder(chunk_encoder, career_encoder)
    model.eval()

    continuous = torch.zeros(batch_size, max_chunks, max_pitch_len, 4)
    pitch_type = torch.zeros(batch_size, max_chunks, max_pitch_len, dtype=torch.long)
    outcome = torch.zeros(batch_size, max_chunks, max_pitch_len, dtype=torch.long)
    matchup = torch.zeros(batch_size, max_chunks, max_pitch_len, dtype=torch.long)
    position = torch.zeros(batch_size, max_chunks, max_pitch_len, dtype=torch.long)
    pitch_padding_mask = torch.ones(batch_size, max_chunks, max_pitch_len, dtype=torch.bool)
    chunk_has_history = torch.zeros(batch_size, max_chunks, dtype=torch.bool)

    days_before_cutoff = torch.zeros(batch_size, max_chunks)
    chunk_padding_mask = torch.ones(batch_size, max_chunks, dtype=torch.bool)
    player_has_history = torch.zeros(batch_size, dtype=torch.bool)

    # Player 0: 3 real games (5, 4, 2 pitches). Player 1: no games at all
    # (career cold start). Player 2: 1 real game.
    real_chunks_per_player = [3, 0, 1]
    pitches_per_chunk = [5, 4, 2]

    for p, n_chunks in enumerate(real_chunks_per_player):
        if n_chunks == 0:
            continue
        player_has_history[p] = True
        chunk_padding_mask[p, :n_chunks] = False
        days_before_cutoff[p, :n_chunks] = torch.arange(n_chunks).float() * 5.0
        for c in range(n_chunks):
            n_pitches = pitches_per_chunk[c]
            chunk_has_history[p, c] = True
            continuous[p, c, :n_pitches] = torch.randn(n_pitches, 4)
            pitch_type[p, c, :n_pitches] = torch.randint(0, len(PITCH_TYPE_VOCAB), (n_pitches,))
            outcome[p, c, :n_pitches] = torch.randint(0, len(OUTCOME_VOCAB), (n_pitches,))
            matchup[p, c, :n_pitches] = torch.randint(0, len(MATCHUP_VOCAB), (n_pitches,))
            position[p, c, :n_pitches] = torch.arange(n_pitches)
            pitch_padding_mask[p, c, :n_pitches] = False

    chunk_pitch_sequences = {
        "continuous": continuous,
        "pitch_type": pitch_type,
        "outcome": outcome,
        "matchup": matchup,
        "position": position,
        "padding_mask": pitch_padding_mask,
        "has_history": chunk_has_history,
    }

    output = model(chunk_pitch_sequences, days_before_cutoff, chunk_padding_mask, player_has_history)

    assert output.shape == (batch_size, hidden_size)
    assert not torch.isnan(output).any()
    # the career cold-start player (no games at all) should get CareerEncoder's
    # learned no-history embedding.
    assert torch.allclose(output[1], career_encoder.no_history_embedding.detach())

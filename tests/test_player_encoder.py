import torch

from src.data.sequence_dataset import MATCHUP_VOCAB, OUTCOME_VOCAB, PITCH_TYPE_VOCAB
from src.models.player_encoder import PlayerEncoder, PlayerEncoderConfig


def _small_config() -> PlayerEncoderConfig:
    return PlayerEncoderConfig(
        hidden_size=16, num_layers=2, num_heads=2, dropout=0.0, feedforward_dim=32, max_seq_len=10
    )


def _make_batch(lengths: list[int]):
    """Build a padded batch (padded to the longest length in `lengths`) plus
    padding_mask and has_history, mimicking what a collate_fn over
    PlayerPitchSequenceDataset samples would produce."""
    batch_size = len(lengths)
    max_len = max(max(lengths), 1)  # keep at least 1 column even if all-empty

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


def test_output_shape_for_mixed_batch_with_empty_sequences():
    model = PlayerEncoder(_small_config())
    model.eval()

    batch = _make_batch([5, 0, 3, 0, 1])
    output = model(*batch)

    assert output.shape == (5, 16)


def test_zero_history_rows_get_the_learned_no_history_embedding():
    model = PlayerEncoder(_small_config())
    model.eval()

    continuous, pitch_type, outcome, matchup, position, padding_mask, has_history = _make_batch([4, 0, 2, 0])
    output = model(continuous, pitch_type, outcome, matchup, position, padding_mask, has_history)

    expected = model.no_history_embedding.detach()
    assert torch.allclose(output[1], expected)
    assert torch.allclose(output[3], expected)
    # sanity: the real-history rows should not just coincidentally match it
    assert not torch.allclose(output[0], expected)
    assert not torch.allclose(output[2], expected)


def test_all_history_batch_runs_without_zero_length_edge_case():
    model = PlayerEncoder(_small_config())
    model.eval()

    batch = _make_batch([5, 3, 7])
    output = model(*batch)
    assert output.shape == (3, 16)


def test_all_empty_batch_skips_transformer_entirely():
    model = PlayerEncoder(_small_config())
    model.eval()

    batch = _make_batch([0, 0, 0])
    output = model(*batch)

    assert output.shape == (3, 16)
    expected = model.no_history_embedding.detach()
    for row in output:
        assert torch.allclose(row, expected)


def test_gradients_flow_through_both_branches():
    model = PlayerEncoder(_small_config())
    batch = _make_batch([4, 0, 2])

    output = model(*batch)
    # Not output.sum(): with LayerNorm's default uniform gamma, the normalized
    # values are mean-zero across the hidden dim, so their sum is ~constant
    # regardless of upstream weights and the gradient vanishes by construction.
    (output**2).sum().backward()

    assert model.no_history_embedding.grad is not None
    assert not torch.all(model.no_history_embedding.grad == 0)
    assert model.continuous_proj.weight.grad is not None
    assert not torch.all(model.continuous_proj.weight.grad == 0)

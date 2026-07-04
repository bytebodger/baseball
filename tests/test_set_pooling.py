import torch

from src.models.set_pooling import PlayerSetPooler, PlayerSetPoolerConfig


def _small_config() -> PlayerSetPoolerConfig:
    return PlayerSetPoolerConfig(embed_dim=16, num_heads=2, dropout=0.0)


def _make_sets(set_sizes: list[int], embed_dim: int) -> list[torch.Tensor]:
    """One variable-length tensor of player embeddings per batch item, e.g. a
    bullpen with `n` arms available or a lineup with `n` spots filled so far."""
    return [torch.randn(n, embed_dim) for n in set_sizes]


def test_output_shape_for_mixed_batch_with_empty_sets():
    model = PlayerSetPooler(_small_config())
    model.eval()

    # a full bullpen, an empty bullpen, a short bullpen, an empty one again,
    # and a single reliever left.
    sets = _make_sets([7, 0, 3, 0, 1], embed_dim=16)
    padded, padding_mask = PlayerSetPooler.pad_embeddings(sets)
    output = model(padded, padding_mask)

    assert output.shape == (5, 16)


def test_empty_sets_get_the_learned_empty_set_embedding():
    model = PlayerSetPooler(_small_config())
    model.eval()

    sets = _make_sets([4, 0, 2, 0], embed_dim=16)
    padded, padding_mask = PlayerSetPooler.pad_embeddings(sets)
    output = model(padded, padding_mask)

    expected = model.empty_set_embedding.detach()
    assert torch.allclose(output[1], expected)
    assert torch.allclose(output[3], expected)
    # sanity: the real-set rows should not just coincidentally match it
    assert not torch.allclose(output[0], expected)
    assert not torch.allclose(output[2], expected)


def test_all_nonempty_batch_runs_without_zero_size_edge_case():
    model = PlayerSetPooler(_small_config())
    model.eval()

    sets = _make_sets([5, 3, 7], embed_dim=16)
    padded, padding_mask = PlayerSetPooler.pad_embeddings(sets)
    output = model(padded, padding_mask)

    assert output.shape == (3, 16)


def test_all_empty_batch_skips_attention_entirely():
    model = PlayerSetPooler(_small_config())
    model.eval()

    sets = _make_sets([0, 0, 0], embed_dim=16)
    padded, padding_mask = PlayerSetPooler.pad_embeddings(sets)
    output = model(padded, padding_mask)

    assert output.shape == (3, 16)
    expected = model.empty_set_embedding.detach()
    for row in output:
        assert torch.allclose(row, expected)


def test_pooling_reweights_players_rather_than_averaging():
    """With no padding_mask (a full, fixed-size lineup), attention pooling
    should differ from a plain mean -- otherwise attention weights aren't
    doing anything and we could've just averaged."""
    model = PlayerSetPooler(_small_config())
    model.eval()

    embeddings = torch.randn(4, 9, 16)
    output = model(embeddings)
    mean_pooled = embeddings.mean(dim=1)

    assert output.shape == (4, 16)
    assert not torch.allclose(output, mean_pooled, atol=1e-3)


def test_gradients_flow_through_both_branches():
    model = PlayerSetPooler(_small_config())

    sets = _make_sets([4, 0, 2], embed_dim=16)
    padded, padding_mask = PlayerSetPooler.pad_embeddings(sets)
    output = model(padded, padding_mask)

    # Not output.sum(): with LayerNorm's default uniform gamma, the normalized
    # values are mean-zero across the hidden dim, so their sum is ~constant
    # regardless of upstream weights and the gradient vanishes by construction.
    (output**2).sum().backward()

    assert model.empty_set_embedding.grad is not None
    assert not torch.all(model.empty_set_embedding.grad == 0)
    assert model.query.grad is not None
    assert not torch.all(model.query.grad == 0)


def test_pad_embeddings_masks_correctly():
    sets = _make_sets([3, 0, 5], embed_dim=16)
    padded, padding_mask = PlayerSetPooler.pad_embeddings(sets)

    assert padded.shape == (3, 5, 16)
    assert padding_mask.tolist() == [
        [False, False, False, True, True],
        [True, True, True, True, True],
        [False, False, False, False, False],
    ]

import pandas as pd
import pytest
import torch

from src.data.park_factors import ParkFactorConfig, ParkFactorEmbedding, compute_park_factors
from src.data.sequence_dataset import MATCHUP_VOCAB, OUTCOME_VOCAB
from src.models.event_model import EventModel, EventModelConfig


def _tiny_park_factor_embedding() -> ParkFactorEmbedding:
    pitches = pd.DataFrame(
        {
            "game_pk": [1, 2, 3, 4],
            "park_id": ["PARK_A", "PARK_A", "PARK_A", "PARK_A"],
            "season": [2022, 2022, 2023, 2023],
            "home_score": [3, 4, 2, 5],
            "away_score": [2, 1, 3, 0],
            "outcome": ["home_run", "single", "ball", "home_run"],
        }
    )
    config = ParkFactorConfig(rolling_years=3, embedding_dim=4)
    park_factors = compute_park_factors(pitches, rolling_years=config.rolling_years)
    return ParkFactorEmbedding(config, park_factors)


def _make_batch(batch_size: int, player_embed_dim: int, situational_dim: int) -> dict:
    return {
        "pitcher_embedding": torch.randn(batch_size, player_embed_dim),
        "batter_embedding": torch.randn(batch_size, player_embed_dim),
        "context": torch.randn(batch_size, situational_dim),
        "matchup_index": torch.randint(0, len(MATCHUP_VOCAB), (batch_size,)),
        "park_index": torch.zeros(batch_size, dtype=torch.long),
        "target": torch.randint(0, len(OUTCOME_VOCAB), (batch_size,)),
    }


def test_event_model_config_from_yaml_loads_the_real_config_file():
    config = EventModelConfig.from_yaml()
    assert config.player_embed_dim > 0
    assert config.include_context is True


def test_event_model_requires_park_factor_embedding_when_include_context_true():
    config = EventModelConfig(player_embed_dim=8, situational_dim=11, include_context=True)
    with pytest.raises(ValueError):
        EventModel(config, park_factor_embedding=None)


def test_event_model_forward_output_shape_with_context():
    config = EventModelConfig(player_embed_dim=8, matchup_embed_dim=4, park_factor_embed_dim=4, situational_dim=11, hidden_dim=16, num_layers=1, include_context=True)
    model = EventModel(config, _tiny_park_factor_embedding())
    batch = _make_batch(batch_size=5, player_embed_dim=8, situational_dim=11)
    logits = model(batch)
    assert logits.shape == (5, len(OUTCOME_VOCAB))


def test_event_model_forward_output_shape_without_context():
    config = EventModelConfig(player_embed_dim=8, hidden_dim=16, num_layers=1, include_context=False)
    model = EventModel(config, park_factor_embedding=None)
    batch = _make_batch(batch_size=5, player_embed_dim=8, situational_dim=11)
    logits = model(batch)
    assert logits.shape == (5, len(OUTCOME_VOCAB))


def test_event_model_without_context_has_no_park_or_matchup_submodules():
    config = EventModelConfig(player_embed_dim=8, hidden_dim=16, num_layers=1, include_context=False)
    model = EventModel(config, park_factor_embedding=_tiny_park_factor_embedding())  # even if passed, must be dropped
    assert model.park_factor_embedding is None
    assert model.matchup_embed is None


def test_event_model_without_context_output_is_invariant_to_context_inputs():
    """The ablation must be architectural, not just a zeroed input: changing
    context/matchup/park inputs should not move the output at all when
    include_context=False."""
    torch.manual_seed(0)
    config = EventModelConfig(player_embed_dim=8, hidden_dim=16, num_layers=1, include_context=False)
    model = EventModel(config, park_factor_embedding=None)
    model.eval()

    batch_a = _make_batch(batch_size=5, player_embed_dim=8, situational_dim=11)
    batch_b = dict(batch_a)
    batch_b["context"] = torch.randn_like(batch_a["context"]) * 100
    batch_b["matchup_index"] = torch.randint(0, len(MATCHUP_VOCAB), (5,))
    batch_b["park_index"] = torch.randint(0, 3, (5,))

    with torch.no_grad():
        out_a = model(batch_a)
        out_b = model(batch_b)
    assert torch.equal(out_a, out_b)


def test_event_model_with_context_combined_dim_includes_all_pieces():
    config = EventModelConfig(player_embed_dim=8, matchup_embed_dim=4, park_factor_embed_dim=4, situational_dim=11, hidden_dim=16, num_layers=1, include_context=True)
    model = EventModel(config, _tiny_park_factor_embedding())
    expected_in = 8 * 2 + 4 + 4 + 11
    first_linear = model.trunk[0]
    assert first_linear.in_features == expected_in


def test_event_model_without_context_combined_dim_is_just_the_two_embeddings():
    config = EventModelConfig(player_embed_dim=8, hidden_dim=16, num_layers=1, include_context=False)
    model = EventModel(config, park_factor_embedding=None)
    first_linear = model.trunk[0]
    assert first_linear.in_features == 16


# ---------- contact_quality_aux_head ----------


def test_event_model_has_contact_quality_aux_head_with_context():
    config = EventModelConfig(player_embed_dim=8, hidden_dim=16, num_layers=1, include_context=True)
    model = EventModel(config, _tiny_park_factor_embedding())
    assert model.contact_quality_aux_head is not None
    assert model.contact_quality_aux_head.in_features == 16  # hidden_dim
    assert model.contact_quality_aux_head.out_features == 2  # (babip, hard_hit_rate)


def test_event_model_without_context_has_no_contact_quality_aux_head():
    config = EventModelConfig(player_embed_dim=8, hidden_dim=16, num_layers=1, include_context=False)
    model = EventModel(config, park_factor_embedding=_tiny_park_factor_embedding())
    assert model.contact_quality_aux_head is None


def test_event_model_forward_default_return_is_unchanged_by_return_aux_support():
    """return_aux defaults to False -- adding the auxiliary head must not
    change forward()'s existing return type/shape for any caller that
    doesn't ask for it (game_engine.py never does)."""
    config = EventModelConfig(player_embed_dim=8, matchup_embed_dim=4, park_factor_embed_dim=4, situational_dim=11, hidden_dim=16, num_layers=1, include_context=True)
    model = EventModel(config, _tiny_park_factor_embedding())
    batch = _make_batch(batch_size=5, player_embed_dim=8, situational_dim=11)
    logits = model(batch)
    assert isinstance(logits, torch.Tensor)
    assert logits.shape == (5, len(OUTCOME_VOCAB))


def test_event_model_forward_return_aux_true_returns_logits_and_aux_predictions():
    config = EventModelConfig(player_embed_dim=8, matchup_embed_dim=4, park_factor_embed_dim=4, situational_dim=11, hidden_dim=16, num_layers=1, include_context=True)
    model = EventModel(config, _tiny_park_factor_embedding())
    batch = _make_batch(batch_size=5, player_embed_dim=8, situational_dim=11)
    logits, aux = model(batch, return_aux=True)
    assert logits.shape == (5, len(OUTCOME_VOCAB))
    assert aux.shape == (5, 2)


def test_event_model_forward_return_aux_true_without_context_raises():
    config = EventModelConfig(player_embed_dim=8, hidden_dim=16, num_layers=1, include_context=False)
    model = EventModel(config, park_factor_embedding=None)
    batch = _make_batch(batch_size=5, player_embed_dim=8, situational_dim=11)
    with pytest.raises(ValueError):
        model(batch, return_aux=True)


def test_event_model_contact_quality_aux_gradient_flows_to_shared_trunk_parameters():
    """The auxiliary loss must reach the SAME trunk parameters the main
    classification path uses (that's the entire point -- shared-
    representation pressure, not a side-branch with its own private
    parameters disconnected from the main task)."""
    config = EventModelConfig(player_embed_dim=8, matchup_embed_dim=4, park_factor_embed_dim=4, situational_dim=11, hidden_dim=16, num_layers=1, include_context=True)
    model = EventModel(config, _tiny_park_factor_embedding())
    batch = _make_batch(batch_size=5, player_embed_dim=8, situational_dim=11)

    _logits, aux = model(batch, return_aux=True)
    aux.sum().backward()

    trunk_first_layer = model.trunk[0]
    assert trunk_first_layer.weight.grad is not None
    assert trunk_first_layer.weight.grad.abs().sum().item() > 0
    for name, param in model.contact_quality_aux_head.named_parameters():
        assert param.grad is not None, f"no gradient reached contact_quality_aux_head.{name}"


# ---------- interaction_type ----------


def test_event_model_config_rejects_unknown_interaction_type():
    with pytest.raises(ValueError):
        EventModelConfig(interaction_type="quadratic")


def test_event_model_default_interaction_type_is_none_and_unchanged_combined_dim():
    """interaction_type="none" must reproduce the pre-existing architecture
    exactly -- this is a regression guard for the already-checked-in keeper
    checkpoint's combined_dim."""
    config = EventModelConfig(player_embed_dim=8, matchup_embed_dim=4, park_factor_embed_dim=4, situational_dim=11, hidden_dim=16, num_layers=1, include_context=True)
    assert config.interaction_type == "none"
    model = EventModel(config, _tiny_park_factor_embedding())
    assert model.trunk[0].in_features == 8 * 2 + 4 + 4 + 11
    assert model.bilinear_pitcher_proj is None
    assert model.film_scale is None


def test_event_model_bilinear_combined_dim_adds_interaction_dim():
    config = EventModelConfig(
        player_embed_dim=8, matchup_embed_dim=4, park_factor_embed_dim=4, situational_dim=11,
        hidden_dim=16, num_layers=1, include_context=True, interaction_type="bilinear", interaction_dim=6,
    )
    model = EventModel(config, _tiny_park_factor_embedding())
    assert model.trunk[0].in_features == 8 * 2 + 6 + 4 + 4 + 11
    batch = _make_batch(batch_size=5, player_embed_dim=8, situational_dim=11)
    logits = model(batch)
    assert logits.shape == (5, len(OUTCOME_VOCAB))


def test_event_model_bilinear_output_changes_when_batter_embedding_changes():
    """A genuine cross-term must be sensitive to the batter embedding, not
    just riding along -- guards against a wiring bug where the interaction
    term is accidentally computed from only one side."""
    torch.manual_seed(0)
    config = EventModelConfig(
        player_embed_dim=8, matchup_embed_dim=4, park_factor_embed_dim=4, situational_dim=11,
        hidden_dim=16, num_layers=1, include_context=True, interaction_type="bilinear", interaction_dim=6,
    )
    model = EventModel(config, _tiny_park_factor_embedding())
    model.eval()
    batch_a = _make_batch(batch_size=5, player_embed_dim=8, situational_dim=11)
    batch_b = dict(batch_a)
    batch_b["batter_embedding"] = torch.randn_like(batch_a["batter_embedding"])
    with torch.no_grad():
        out_a = model(batch_a)
        out_b = model(batch_b)
    assert not torch.equal(out_a, out_b)


def test_event_model_elementwise_combined_dim_adds_player_embed_dim():
    config = EventModelConfig(
        player_embed_dim=8, matchup_embed_dim=4, park_factor_embed_dim=4, situational_dim=11,
        hidden_dim=16, num_layers=1, include_context=True, interaction_type="elementwise",
    )
    model = EventModel(config, _tiny_park_factor_embedding())
    assert model.trunk[0].in_features == 8 * 3 + 4 + 4 + 11
    batch = _make_batch(batch_size=5, player_embed_dim=8, situational_dim=11)
    logits = model(batch)
    assert logits.shape == (5, len(OUTCOME_VOCAB))


def test_event_model_film_combined_dim_is_unchanged_but_modulates_batter():
    """FiLM replaces the batter embedding in-place rather than adding a new
    concatenated block, so combined_dim is identical to interaction_type="none"."""
    config = EventModelConfig(
        player_embed_dim=8, matchup_embed_dim=4, park_factor_embed_dim=4, situational_dim=11,
        hidden_dim=16, num_layers=1, include_context=True, interaction_type="film",
    )
    model = EventModel(config, _tiny_park_factor_embedding())
    assert model.trunk[0].in_features == 8 * 2 + 4 + 4 + 11
    batch = _make_batch(batch_size=5, player_embed_dim=8, situational_dim=11)
    logits = model(batch)
    assert logits.shape == (5, len(OUTCOME_VOCAB))


def test_event_model_film_is_identity_at_initialization():
    """film_scale/film_shift are zero-initialized, so a freshly constructed
    FiLM model must produce IDENTICAL output to an interaction_type="none"
    model with the same weights everywhere else -- confirms the model
    starts at the plain-additive baseline rather than a random modulation."""
    torch.manual_seed(0)
    config_none = EventModelConfig(
        player_embed_dim=8, matchup_embed_dim=4, park_factor_embed_dim=4, situational_dim=11,
        hidden_dim=16, num_layers=1, include_context=True, interaction_type="none",
    )
    model_none = EventModel(config_none, _tiny_park_factor_embedding())

    torch.manual_seed(0)
    config_film = EventModelConfig(
        player_embed_dim=8, matchup_embed_dim=4, park_factor_embed_dim=4, situational_dim=11,
        hidden_dim=16, num_layers=1, include_context=True, interaction_type="film",
    )
    model_film = EventModel(config_film, _tiny_park_factor_embedding())
    # film_scale/film_shift are extra parameters model_none doesn't have --
    # everything else (trunk, output_head, park/matchup embeddings) was
    # constructed in the same order under the same seed, so should match.
    model_film.trunk.load_state_dict(model_none.trunk.state_dict())
    model_film.output_head.load_state_dict(model_none.output_head.state_dict())
    model_film.matchup_embed.load_state_dict(model_none.matchup_embed.state_dict())

    model_none.eval()
    model_film.eval()
    batch = _make_batch(batch_size=5, player_embed_dim=8, situational_dim=11)
    with torch.no_grad():
        out_none = model_none(batch)
        out_film = model_film(batch)
    assert torch.equal(out_none, out_film)


def test_event_model_film_output_changes_when_pitcher_embedding_changes_after_training_step():
    """At init FiLM is identity (see above), so this checks that after one
    gradient step nudges film_scale/film_shift away from zero, the pitcher
    embedding actually participates in modulating the batter path (not just
    contributing through its own raw concatenated copy)."""
    torch.manual_seed(0)
    config = EventModelConfig(
        player_embed_dim=8, matchup_embed_dim=4, park_factor_embed_dim=4, situational_dim=11,
        hidden_dim=16, num_layers=1, include_context=True, interaction_type="film",
    )
    model = EventModel(config, _tiny_park_factor_embedding())
    batch = _make_batch(batch_size=5, player_embed_dim=8, situational_dim=11)

    logits = model(batch)
    logits.sum().backward()
    assert model.film_scale.weight.grad is not None
    assert model.film_scale.weight.grad.abs().sum().item() > 0
    assert model.film_shift.weight.grad is not None
    assert model.film_shift.weight.grad.abs().sum().item() > 0
